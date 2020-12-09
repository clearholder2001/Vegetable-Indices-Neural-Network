import os


# Data
# ------------------------------------------------
RGB_PATH = os.path.join('..', 'Jim', 'dataset', '20meter', 'train_20meter_RGB.npy')
NDVI_PATH = os.path.join('..', 'Jim', 'dataset', '20meter', 'train_20meter_NDVI.npy')

RESAMPLE_MULTIPLE_FACTOR = 9


# Model
# ------------------------------------------------
INPUT_LAYER_DIM = (352, 480, 3)
LEARNING_RATE = 0.0002


# Data Augmentation
# ------------------------------------------------
ENABLE_DATA_AUG = False
DATA_AUG_MULTIPLE_FACTOR = 16
DATA_AUG_BATCH_SIZE = 32


# Train
# ------------------------------------------------
EPOCHS = 100
TRAIN_BATCH_SIZE = 32
