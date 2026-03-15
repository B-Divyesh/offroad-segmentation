# Offroad Semantic Segmentation - Solution

## Overview

Semantic segmentation solution for Duality AI's Offroad Autonomy Segmentation Challenge.
Uses a **DeepLabV3+** decoder with a **ConvNeXt-V2-Large** encoder (~198M params) via
`segmentation_models_pytorch`, trained on synthetic desert terrain imagery.

**Best Validation mIoU: 0.6895**

## Architecture

- **Encoder**: ConvNeXt-V2-Large (`tu-convnextv2_large`), ImageNet pretrained via timm
- **Decoder**: DeepLabV3+ (via `segmentation_models_pytorch`)
- **Training**: Train+val combined (3,174 images) with very aggressive augmentation
- **Loss**: CrossEntropy (label_smoothing=0.05) + Dice + 0.5 * Lovász-Softmax
- **TTA**: Horizontal flip averaging at inference

## Environment Setup

```bash
# Python 3.10+, PyTorch 2.x with CUDA
pip install torch torchvision
pip install segmentation-models-pytorch
pip install albumentations
pip install timm
pip install numpy pillow tqdm
```

ConvNeXt-V2-Large weights are automatically downloaded from HuggingFace Hub on first run.

## Directory Structure

```
solutions1/
  config.py          # Configuration (paths, hyperparameters, class definitions)
  train.py           # Training script
  test.py            # Evaluation/inference script
  README.md          # This file
  report.md          # Detailed hackathon report
  best_model.pth     # Model checkpoint (download separately, see below)
```

## How to Run

### 1. Update paths in config.py

Edit `config.py` to set `TRAIN_DIR`, `VAL_DIR`, `TEST_DIR` to your local dataset paths.

### 2. Training

```bash
python train.py
```

Paths and hyperparameters are configured at the top of `train.py`:
- `TRAIN_DIR`, `VAL_DIR`: Point to your dataset directories
- `DEVICE`: Set to `cuda:0` (default)
- `BATCH_SIZE`: 3 (for ~19GB VRAM usage)
- `GRAD_ACCUM`: 6 (effective batch size = 18)

Training takes ~6 hours on a single NVIDIA H100 for 120 epochs.

### 3. Testing / Evaluation

```bash
python test.py \
  --checkpoint models/v20/best_model.pth \
  --test-dir data/test/Offroad_Segmentation_testImages
```

Options:
- `--no-tta`: Disable test-time augmentation
- `--output-dir predictions`: Directory for predicted masks

### Expected Output

```
Val mIoU = 0.6895
```

Per-class IoU on validation set:
| Class | IoU |
|-------|-----|
| Background | 0.797 |
| Trees | 0.811 |
| Lush Bushes | 0.539 |
| Dry Grass | 0.650 |
| Dry Bushes | 0.513 |
| Ground Clutter | 0.396 |
| Logs | 0.396 |
| Rocks | 0.498 |
| Landscape | 0.862 |
| Sky | 0.933 |

## Model Checkpoint

The checkpoint file `best_model.pth` is obtained from the training output.
It contains a dict with keys:
- `model_state_dict`: Full model weights
- `epoch`: Training epoch when saved
- `val_iou`: Validation mIoU at save time
- `class_ious`: Per-class IoU breakdown

## Notes

- **Train+Val Combined**: Following hackathon workflow guidance, we train on the full
  train+val dataset (3,174 images) to maximize data utilization.
- **Aggressive Augmentation**: RandomResizedCrop, affine transforms, color jitter,
  blur/noise, shadows -- designed for maximum domain robustness.
- **No test data was used for training** -- test images were only used for evaluation,
  in full compliance with competition rules.
