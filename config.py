"""Configuration for Offroad Semantic Segmentation Model."""

# Dataset
VALUE_MAP = {0: 0, 100: 1, 200: 2, 300: 3, 500: 4, 550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9}
CLASS_NAMES = ['Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
               'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky']
NUM_CLASSES = 10

# Input
IMG_H = 518
IMG_W = 518

# Paths (adjust these for your environment)
TRAIN_DIR = 'data/train/Offroad_Segmentation_Training_Dataset/train'
VAL_DIR = 'data/train/Offroad_Segmentation_Training_Dataset/val'
TEST_DIR = 'data/test/Offroad_Segmentation_testImages'
OUTPUT_DIR = 'models/v13'
LOG_FILE = 'logs/v13_training.log'

# Model
BACKBONE = 'dinov2_vitl14'
EXTRACT_LAYERS = [5, 11, 17, 23]
EMBED_DIM = 1024

# Training - Phase 1 (frozen backbone)
BATCH_SIZE = 8
NUM_EPOCHS_PHASE1 = 20
LEARNING_RATE_PHASE1 = 3e-4
WEIGHT_DECAY = 1e-4

# Training - Phase 2 (LoRA fine-tuning)
NUM_EPOCHS_PHASE2 = 10
LEARNING_RATE_PHASE2 = 1e-4
LORA_RANK = 16
LORA_ALPHA = 32

# Loss
LABEL_SMOOTHING = 0.05

# Inference
SUPPRESS_CLASSES = [0, 5, 6]  # Classes absent from test set
USE_TTA = True  # Horizontal flip TTA
