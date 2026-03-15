"""Test/Inference script for V20b Offroad Semantic Segmentation.

Architecture: DeepLabV3+ with ConvNeXt-V2-Large encoder (segmentation_models_pytorch)

Usage:
    python test.py --checkpoint models/v20/best_model.pth --test-dir data/test/Offroad_Segmentation_testImages
"""
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    print("Please install: pip install albumentations")
    exit(1)

from config import VALUE_MAP, CLASS_NAMES, NUM_CLASSES, IMG_H, IMG_W, ENCODER_NAME

class SegDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.img_dir = os.path.join(data_dir, 'Color_Images')
        self.mask_dir = os.path.join(data_dir, 'Segmentation')
        self.files = sorted(os.listdir(self.img_dir))
        self.transform = transform
        self.has_masks = os.path.isdir(self.mask_dir) and len(os.listdir(self.mask_dir)) > 0

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img = np.array(Image.open(os.path.join(self.img_dir, fname)).convert('RGB'))

        mask = None
        if self.has_masks:
            raw = np.array(Image.open(os.path.join(self.mask_dir, fname)))
            mask = np.zeros(raw.shape[:2], dtype=np.int64)
            for raw_val, cls_id in VALUE_MAP.items():
                mask[raw == raw_val] = cls_id

        if self.transform:
            if mask is not None:
                t = self.transform(image=img, mask=mask)
                img, mask = t['image'], t['mask'].long()
            else:
                t = self.transform(image=img)
                img = t['image']

        return img, mask if mask is not None else torch.tensor(-1), fname

def get_val_transform():
    return A.Compose([
        A.Resize(IMG_H, IMG_W),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def compute_iou(preds, masks, num_classes=10):
    ious = []
    for c in range(num_classes):
        pred_c = (preds == c)
        gt_c = (masks == c)
        inter = (pred_c & gt_c).sum()
        union = (pred_c | gt_c).sum()
        ious.append(float(inter) / float(union) if union > 0 else float('nan'))
    valid = [v for v in ious if not np.isnan(v)]
    return np.mean(valid) if valid else 0, ious

def predict_tta(model, imgs):
    """TTA: original + horizontal flip."""
    with torch.no_grad():
        l1 = model(imgs)
        l2 = torch.flip(model(torch.flip(imgs, [3])), [3])
        return (F.softmax(l1, 1) + F.softmax(l2, 1)) / 2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--test-dir', required=True, help='Path to test data directory')
    parser.add_argument('--output-dir', default='predictions', help='Output directory for predictions')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--no-tta', action='store_true', help='Disable TTA')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create model
    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER_NAME,
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
    )

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"  Epoch: {ckpt.get('epoch', '?')}, Val IoU: {ckpt.get('val_iou', '?')}")
    else:
        model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()

    # Dataset
    transform = get_val_transform()
    dataset = SegDataset(args.test_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    print(f"Test images: {len(dataset)}")

    os.makedirs(args.output_dir, exist_ok=True)

    all_preds = []
    all_masks = []
    all_fnames = []

    with torch.no_grad():
        for imgs, masks, fnames in loader:
            imgs = imgs.to(device)

            if args.no_tta:
                logits = model(imgs)
                probs = F.softmax(logits, 1)
            else:
                probs = predict_tta(model, imgs)

            preds = probs.argmax(1).cpu().numpy()

            for i, fname in enumerate(fnames):
                pred = preds[i]
                # Save prediction as PNG
                pred_img = Image.fromarray(pred.astype(np.uint8))
                pred_img.save(os.path.join(args.output_dir, fname))
                all_fnames.append(fname)

            all_preds.append(preds)
            if masks[0].item() != -1:
                all_masks.append(masks.numpy())

    print(f"Saved {len(all_fnames)} predictions to {args.output_dir}")

    # If masks available, compute IoU
    if all_masks:
        preds = np.concatenate(all_preds)
        masks = np.concatenate(all_masks)
        mean_iou, class_ious = compute_iou(preds, masks)
        print(f"\nMean IoU: {mean_iou:.4f}")
        for i, name in enumerate(CLASS_NAMES):
            if not np.isnan(class_ious[i]):
                print(f"  {name}: {class_ious[i]:.4f}")

if __name__ == '__main__':
    main()
