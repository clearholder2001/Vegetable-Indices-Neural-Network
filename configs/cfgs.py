import os


# Data
# ------------------------------------------------
TRAIN_RGB_PATH = os.path.join('..', 'Jim', 'dataset', '20meter', 'train_20meter_RGB.npy')
TRAIN_NDVI_PATH = os.path.join('..', 'Jim', 'dataset', '20meter', 'train_20meter_NDVI.npy')
TEST_RGB_PATH = os.path.join('..', 'Jim', 'dataset', 'testing', 'test_15meter_RGB.npy')
TEST_NDVI_PATH = os.path.join('..', 'Jim', 'dataset', 'testing', 'test_15meter_NDVI.npy')

RESAMPLE_MULTIPLE_FACTOR = 9


# Model
# ------------------------------------------------
INPUT_LAYER_DIM = (352, 480, 3)
LEARNING_RATE = 0.0002


# Data Augmentation
# ------------------------------------------------
ENABLE_DATA_AUG = True
DATA_AUG_BATCH_SIZE = 32


# Train
# ------------------------------------------------
EPOCHS = 100
TRAIN_BATCH_SIZE = 32
VAL_SPLIT = 0.1
