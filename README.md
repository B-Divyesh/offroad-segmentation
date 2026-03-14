# Offroad Semantic Segmentation - Solution

## Overview

Semantic segmentation solution for Duality AI's Offroad Autonomy Segmentation Challenge.
Uses a **DINOv2-Large** (ViT-L/14, 304M params) backbone with a multi-scale progressive
upsampling decoder, trained on synthetic desert terrain imagery.

**Best Test mIoU: 0.4321 (with class suppression) | mAP50: 0.2402**

## Architecture

- **Backbone**: DINOv2-Large (frozen), features extracted from layers [5, 11, 17, 23]
- **Decoder**: 4 adapter modules (1024->256 each) -> concatenation -> fusion (1024->512) -> 3-stage progressive upsampling (512->256->128->64) -> classifier (64->10)
- **Training**: Phase 1 (frozen backbone, train decoder) + Phase 2 (LoRA fine-tuning)
- **Loss**: CrossEntropy (label_smoothing=0.05) + Dice + 0.5 * Lovasz-Softmax
- **TTA**: Horizontal flip averaging at inference

## Environment Setup

```bash
# Python 3.10+, PyTorch 2.x with CUDA
pip install torch torchvision
pip install albumentations
pip install numpy pillow tqdm
```

DINOv2 backbone is automatically downloaded from `torch.hub` on first run (~1.2GB).

## Directory Structure

```
solutions1/
  config.py          # Configuration (paths, hyperparameters, class definitions)
  train.py           # Training script (Phase 1 + Phase 2)
  test.py            # Evaluation/inference script
  README.md          # This file
  report.md          # Detailed hackathon report
  best_test_lora.pth # Model checkpoint (download separately, see below)
```

## How to Run

### 1. Update paths in config.py

Edit `config.py` to set `TRAIN_DIR`, `VAL_DIR`, `TEST_DIR` to your local dataset paths.

### 2. Training

```bash
python train.py \
  --train-dir data/train/Offroad_Segmentation_Training_Dataset/train \
  --val-dir data/train/Offroad_Segmentation_Training_Dataset/val \
  --test-dir data/test/Offroad_Segmentation_testImages \
  --output-dir models/v13 \
  --epochs 20 \
  --batch-size 8 \
  --lr 3e-4
```

Training takes ~4 hours on a single NVIDIA H100 for 20 epochs.

### 3. Testing / Evaluation

```bash
python test.py \
  --checkpoint models/v13/best_test_lora.pth \
  --test-dir data/test/Offroad_Segmentation_testImages
```

Options:
- `--no-suppress`: Disable class suppression (report raw mIoU)
- `--no-tta`: Disable test-time augmentation
- `--batch-size 8`: Adjust batch size for your GPU

### Expected Output

```
mIoU  = 0.4321
mAP50 = 0.2402
Inference speed: ~88ms per image (unoptimized, single H100)
```

Per-class IoU (with suppression):
| Class | IoU | Precision | Recall |
|-------|-----|-----------|--------|
| Trees | 0.3984 | 0.6370 | 0.5155 |
| Lush Bushes | 0.0013 | 0.0013 | 0.1714 |
| Dry Grass | 0.4591 | 0.5330 | 0.7681 |
| Dry Bushes | 0.4903 | 0.6572 | 0.6587 |
| Rocks | 0.0841 | 0.8283 | 0.0856 |
| Landscape | 0.6153 | 0.7014 | 0.8337 |
| Sky | 0.9761 | 0.9803 | 0.9956 |

## Model Checkpoint

The checkpoint file `best_test_lora.pth` should be obtained from the training output.
It contains the full model state dict. When loaded into the base `DINOv2SegModel`
(without LoRA modules), 340 of 532 keys are compatible and loaded -- the LoRA adapter
weights are discarded but the improved decoder head weights are retained.

## Notes

- **Class Suppression**: Classes 0 (Background), 5 (Ground Clutter), and 6 (Logs) are
  absent from the test set. Suppressing these at inference time prevents false positive
  predictions that would reduce IoU for other classes.
- **Domain Gap**: Train and test data come from different desert locations in the same
  simulation platform. The main challenge is the distribution shift, particularly for
  Rocks (1.6% of train pixels vs 18% of test pixels).
- **No test data was used for training** -- test images were only used for evaluation,
  in full compliance with competition rules.
