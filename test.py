"""Test/Inference script for Offroad Semantic Segmentation.

Usage:
    python test.py --checkpoint models/v13/best_test_lora.pth --test-dir data/test/Offroad_Segmentation_testImages
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import os
import time
from torch.utils.data import Dataset, DataLoader
from PIL import Image

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    print("Please install: pip install albumentations")
    exit(1)

from config import (VALUE_MAP, CLASS_NAMES, NUM_CLASSES, IMG_H, IMG_W,
                     SUPPRESS_CLASSES, EXTRACT_LAYERS)


class SegDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.img_dir = os.path.join(data_dir, 'Color_Images')
        self.mask_dir = os.path.join(data_dir, 'Segmentation')
        self.files = sorted(os.listdir(self.img_dir))
        self.has_masks = os.path.exists(self.mask_dir)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img = np.array(Image.open(os.path.join(self.img_dir, fname)).convert('RGB'))

        if self.has_masks:
            raw = np.array(Image.open(os.path.join(self.mask_dir, fname)))
            mask = np.zeros(raw.shape[:2], dtype=np.int64)
            for raw_val, cls_id in VALUE_MAP.items():
                mask[raw == raw_val] = cls_id
        else:
            mask = np.zeros(img.shape[:2], dtype=np.int64)

        if self.transform:
            t = self.transform(image=img, mask=mask)
            img, mask = t['image'], t['mask']
            return img, mask.long(), fname
        return img, mask, fname


class DINOv2SegModel(nn.Module):
    """DINOv2-Large backbone with multi-scale progressive upsampling decoder."""

    def __init__(self, num_classes=10):
        super().__init__()
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


def compute_iou(pred, target, num_classes=NUM_CLASSES):
    """Compute per-class IoU matching the competition's evaluation method."""
    pred = pred.view(-1)
    target = target.view(-1)
    iou_per_class = []
    for class_id in range(num_classes):
        pred_inds = pred == class_id
        target_inds = target == class_id
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        if union == 0:
            iou_per_class.append(float('nan'))
        else:
            iou_per_class.append((intersection / union).cpu().numpy())
    return np.nanmean(iou_per_class), iou_per_class


def evaluate(model, loader, device, suppress_classes=None, use_tta=True):
    """Evaluate model on test set. Returns confusion matrix."""
    model.eval()
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    with torch.no_grad():
        for imgs, masks, fnames in loader:
            imgs = imgs.to(device)
            logits = model(imgs)

            if use_tta:
                logits_flip = torch.flip(model(torch.flip(imgs, [3])), [3])
                probs = (F.softmax(logits, 1) + F.softmax(logits_flip, 1)) / 2
            else:
                probs = F.softmax(logits, 1)

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

    return confusion


def print_results(confusion):
    """Print IoU, precision, recall, and mAP50 from confusion matrix."""
    print(f"\n{'Class':>15s} {'IoU':>8s} {'Precision':>10s} {'Recall':>8s} {'GT_pixels':>12s}")
    print("-" * 58)

    valid_ious = []
    aps = []

    for c in range(NUM_CLASSES):
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp
        union = tp + fp + fn
        gt_pix = confusion[c, :].sum()

        iou = float(tp) / float(union) if union > 0 else float('nan')
        prec = float(tp) / float(tp + fp) if (tp + fp) > 0 else float('nan')
        rec = float(tp) / float(tp + fn) if (tp + fn) > 0 else float('nan')

        iou_s = f"{iou:.4f}" if not np.isnan(iou) else "  N/A"
        prec_s = f"{prec:.4f}" if not np.isnan(prec) else "     N/A"
        rec_s = f"{rec:.4f}" if not np.isnan(rec) else "  N/A"
        print(f"{CLASS_NAMES[c]:>15s} {iou_s:>8s} {prec_s:>10s} {rec_s:>8s} {gt_pix:>12d}")

        if not np.isnan(iou):
            valid_ious.append(iou)
        if gt_pix > 0:
            aps.append(prec if iou >= 0.5 else 0.0)

    miou = np.mean(valid_ious) if valid_ious else 0
    map50 = np.mean(aps) if aps else 0
    print(f"\n  mIoU  = {miou:.4f}")
    print(f"  mAP50 = {map50:.4f}")
    return miou, map50


def main():
    parser = argparse.ArgumentParser(description='Test Offroad Segmentation Model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--test-dir', type=str, required=True, help='Path to test data directory')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--no-suppress', action='store_true', help='Disable class suppression')
    parser.add_argument('--no-tta', action='store_true', help='Disable test-time augmentation')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    print("Loading model...")
    model = DINOv2SegModel(num_classes=NUM_CLASSES)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state = ckpt['model_state_dict']
    model_dict = model.state_dict()
    compatible = {k: v for k, v in state.items()
                  if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(compatible)
    model.load_state_dict(model_dict)
    model = model.to(device).eval()
    print(f"  Loaded {len(compatible)}/{len(state)} compatible keys")
    info = {k: v for k, v in ckpt.items() if k != 'model_state_dict'}
    print(f"  Checkpoint info: {info}")

    # Dataset
    transform = A.Compose([
        A.Resize(IMG_H, IMG_W),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    test_ds = SegDataset(args.test_dir, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=4)
    print(f"  Test set: {len(test_ds)} images")

    # Evaluate
    suppress = None if args.no_suppress else SUPPRESS_CLASSES
    use_tta = not args.no_tta

    print(f"\nEvaluating (TTA={'on' if use_tta else 'off'}, "
          f"suppress={suppress or 'none'})...")
    t0 = time.time()
    confusion = evaluate(model, test_loader, device, suppress, use_tta)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    miou, map50 = print_results(confusion)

    # Inference speed benchmark
    print("\nInference speed benchmark...")
    dummy = torch.randn(1, 3, IMG_H, IMG_W, device=device)
    for _ in range(3):
        with torch.no_grad():
            model(dummy)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    n_runs = 20
    for _ in range(n_runs):
        with torch.no_grad():
            model(dummy)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ms_per_img = (time.time() - t0) / n_runs * 1000
    print(f"  {ms_per_img:.1f}ms per image (batch_size=1)")


if __name__ == '__main__':
    main()
