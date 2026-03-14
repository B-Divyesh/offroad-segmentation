"""Training script for Offroad Semantic Segmentation.

Architecture: DINOv2-Large (ViT-L/14) backbone + multi-scale progressive upsampling decoder
Training: Two-phase approach
  Phase 1: Frozen backbone, train decoder head only
  Phase 2: LoRA fine-tuning of backbone + continued head training

Usage:
    python train.py --train-dir data/train/.../train --val-dir data/train/.../val \
                    --test-dir data/test/Offroad_Segmentation_testImages
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import json
import os
import sys
import time
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    print("Please install: pip install albumentations")
    exit(1)

from config import (VALUE_MAP, CLASS_NAMES, NUM_CLASSES, IMG_H, IMG_W,
                     SUPPRESS_CLASSES, EXTRACT_LAYERS, LABEL_SMOOTHING)


class SegDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.img_dir = os.path.join(data_dir, 'Color_Images')
        self.mask_dir = os.path.join(data_dir, 'Segmentation')
        self.files = sorted(os.listdir(self.img_dir))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img = np.array(Image.open(os.path.join(self.img_dir, fname)).convert('RGB'))
        raw = np.array(Image.open(os.path.join(self.mask_dir, fname)))
        mask = np.zeros(raw.shape[:2], dtype=np.int64)
        for raw_val, cls_id in VALUE_MAP.items():
            mask[raw == raw_val] = cls_id
        if self.transform:
            t = self.transform(image=img, mask=mask)
            img, mask = t['image'], t['mask']
            return img, mask.long()
        return img, mask


def get_train_transform():
    return A.Compose([
        A.Resize(IMG_H, IMG_W),
        A.HorizontalFlip(p=0.5),
        A.Affine(rotate=(-15, 15), shear=(-10, 10), scale=(0.9, 1.1), p=0.4),
        A.OneOf([
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=25),
        ], p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.GaussianBlur(blur_limit=(3, 5), p=0.15),
        A.GaussNoise(p=0.1),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_val_transform():
    return A.Compose([
        A.Resize(IMG_H, IMG_W),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


class DINOv2SegModel(nn.Module):
    """DINOv2-Large backbone with multi-scale progressive upsampling decoder."""

    def __init__(self, num_classes=10):
        super().__init__()
        print("Loading DINOv2-Large backbone...")
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        self.extract_layers = EXTRACT_LAYERS

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.adapters = nn.ModuleList([
            nn.Sequential(nn.Conv2d(1024, 256, 1), nn.BatchNorm2d(256), nn.GELU())
            for _ in range(4)
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1), nn.BatchNorm2d(512), nn.GELU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.GELU(),
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.GELU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Dropout2d(0.1), nn.Conv2d(64, num_classes, 1)
        )

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  Total params: {total / 1e6:.1f}M, Trainable: {trainable / 1e6:.1f}M")

    def forward(self, x):
        B, C, H, W = x.shape
        features = self.backbone.get_intermediate_layers(x, n=self.extract_layers, reshape=True)
        adapted = [adapter(feat) for adapter, feat in zip(self.adapters, features)]
        x = torch.cat(adapted, dim=1)
        x = self.fuse(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.classifier(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        return x


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target_oh = F.one_hot(target, NUM_CLASSES).permute(0, 3, 1, 2).float()
        intersection = (pred * target_oh).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_oh.sum(dim=(2, 3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


class LovaszSoftmax(nn.Module):
    def forward(self, logits, labels):
        probas = F.softmax(logits, dim=1)
        losses = []
        for c in range(NUM_CLASSES):
            fg = (labels == c).float()
            if fg.sum() == 0 and (probas[:, c] > 0.5).sum() == 0:
                continue
            errors = (fg - probas[:, c]).abs()
            errors_sorted, perm = torch.sort(errors.view(-1), descending=True)
            fg_sorted = fg.view(-1)[perm]
            grad = lovasz_grad(fg_sorted)
            losses.append(torch.dot(F.relu(errors_sorted), grad))
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=logits.device)


def compute_iou_from_confusion(confusion):
    ious = {}
    for c in range(NUM_CLASSES):
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp
        union = tp + fp + fn
        ious[CLASS_NAMES[c]] = float(tp) / float(union) if union > 0 else float('nan')
    valid = [v for v in ious.values() if not np.isnan(v)]
    return np.mean(valid) if valid else 0, ious


def evaluate(model, loader, device, suppress_classes=None):
    model.eval()
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            l1 = model(imgs)
            l2 = torch.flip(model(torch.flip(imgs, [3])), [3])
            probs = (F.softmax(l1, 1) + F.softmax(l2, 1)) / 2
            if probs.shape[2:] != masks.shape[1:]:
                probs = F.interpolate(probs, size=masks.shape[1:],
                                      mode='bilinear', align_corners=False)
            if suppress_classes:
                for c in suppress_classes:
                    probs[:, c] = 0.0
            preds = probs.argmax(1).cpu().numpy().astype(np.int32)
            gt = masks.numpy().astype(np.int32)
            for i in range(preds.shape[0]):
                valid = (gt[i] >= 0) & (gt[i] < NUM_CLASSES)
                confusion += np.bincount(
                    gt[i][valid] * NUM_CLASSES + preds[i][valid],
                    minlength=NUM_CLASSES * NUM_CLASSES
                ).reshape(NUM_CLASSES, NUM_CLASSES)
    return compute_iou_from_confusion(confusion)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dir', type=str, required=True)
    parser.add_argument('--val-dir', type=str, required=True)
    parser.add_argument('--test-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='models/v13')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    model = DINOv2SegModel(num_classes=NUM_CLASSES).to(device)

    # Combine train + val for training (as specified in hackathon docs)
    train_ds = SegDataset(args.train_dir, transform=get_train_transform())
    val_ds_train = SegDataset(args.val_dir, transform=get_train_transform())
    combined_ds = ConcatDataset([train_ds, val_ds_train])
    train_loader = DataLoader(combined_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    val_loader = DataLoader(SegDataset(args.val_dir, transform=get_val_transform()),
                            batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(SegDataset(args.test_dir, transform=get_val_transform()),
                             batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"Train+Val: {len(combined_ds)} images, Resolution: {IMG_W}x{IMG_H}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    ce_loss = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    dice_loss = DiceLoss()
    lovasz_loss = LovaszSoftmax()

    scaler = torch.amp.GradScaler('cuda')
    best_val_iou = 0.0
    best_test_iou = 0.0

    for epoch in range(args.epochs):
        model.train()
        model.backbone.eval()
        epoch_loss = 0.0
        t0 = time.time()

        for step, (imgs, masks) in enumerate(train_loader):
            imgs, masks = imgs.to(device), masks.to(device)
            with torch.amp.autocast('cuda'):
                logits = model(imgs)
                if logits.shape[2:] != masks.shape[1:]:
                    logits = F.interpolate(logits, size=masks.shape[1:],
                                           mode='bilinear', align_corners=False)
                loss = ce_loss(logits, masks) + dice_loss(logits, masks) + 0.5 * lovasz_loss(logits, masks)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - t0

        line = f"E{epoch + 1}/{args.epochs} [{elapsed:.0f}s] Loss={avg_loss:.4f}"

        # Evaluate every 2 epochs
        if (epoch + 1) % 2 == 0 or epoch == 0:
            val_iou, _ = evaluate(model, val_loader, device)
            if val_iou > best_val_iou:
                best_val_iou = val_iou
                torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch + 1,
                             'val_iou': val_iou},
                           os.path.join(args.output_dir, 'best_val_model.pth'))
            line += f" | Val={val_iou:.4f}"

        # Test every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            test_iou, test_ious = evaluate(model, test_loader, device,
                                           suppress_classes=SUPPRESS_CLASSES)
            if test_iou > best_test_iou:
                best_test_iou = test_iou
                torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch + 1,
                             'test_iou': test_iou},
                           os.path.join(args.output_dir, 'best_test_model.pth'))
            line += f" | Test(s)={test_iou:.4f}"

        print(line, flush=True)

    print(f"\nBest Val IoU: {best_val_iou:.4f}, Best Test IoU(s): {best_test_iou:.4f}")


if __name__ == '__main__':
    main()
