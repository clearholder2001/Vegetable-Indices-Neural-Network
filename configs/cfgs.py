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
MODEL_NAME = 'AE_v0.4.1_swish'
INPUT_LAYER_DIM = (352, 480, 3)
RANDOMCONTRAST_FACTOR = 0.2
L2_REGULAR = 0.001


# Data Augmentation
# ------------------------------------------------
ENABLE_DATA_AUG = False
DATA_AUG_BATCH_SIZE = 16


# Train
# ------------------------------------------------
EPOCHS = 200
TRAIN_BATCH_SIZE = 16
VAL_SPLIT = 0.1
INIT_LEARNING_RATE = 0.001
DECAY_STEPS = 1000
DECAY_RATE = 0.98
STAIRCASE = False
EARLY_STOP_LOSS = 0.02


# ImageDataGenerator
# ------------------------------------------------
DATAGEN_ARGS = dict(
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=20,
    #width_shift_range=0.1,
    #height_shift_range=0.1,
    #brightness_range=(0.5, 1.5),
    #shear_range=0.3,
    zoom_range=0.3,
    #channel_shift_range=0.1,
    #rescale=1/255.,
    #featurewise_center=False,
    #samplewise_center=False,
    #featurewise_std_normalization=False,
    #samplewise_std_normalization=False,
    #zca_whitening=False,
    #zca_epsilon=1e-06,
    fill_mode='reflect',
    #cval=0.0,
    #preprocessing_function=None,
    #data_format=None,
    validation_split=VAL_SPLIT,
    #dtype=None,
)
