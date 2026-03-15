"""Configuration for Offroad Semantic Segmentation Model - V20b."""

# Dataset
VALUE_MAP = {0: 0, 100: 1, 200: 2, 300: 3, 500: 4, 550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9}
CLASS_NAMES = ['Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
               'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky']
NUM_CLASSES = 10

# Input
IMG_H = 768
IMG_W = 768

# Paths (adjust these for your environment)
TRAIN_DIR = 'data/train/Offroad_Segmentation_Training_Dataset/train'
VAL_DIR = 'data/train/Offroad_Segmentation_Training_Dataset/val'
TEST_DIR = 'data/test/Offroad_Segmentation_testImages'
OUTPUT_DIR = 'models/v20'
LOG_FILE = 'logs/v20_training.log'

# Model
ENCODER_NAME = 'tu-convnextv2_large'
ARCHITECTURE = 'DeepLabV3Plus'

# Training
BATCH_SIZE = 3
GRAD_ACCUM = 6  # Effective batch = 18
NUM_EPOCHS = 120
ENCODER_LR = 3e-5
DECODER_LR = 1e-4
WEIGHT_DECAY = 1e-4

# Loss
LABEL_SMOOTHING = 0.05
LOVASZ_WEIGHT = 0.5

# Inference
USE_TTA = True  # Horizontal flip TTA
