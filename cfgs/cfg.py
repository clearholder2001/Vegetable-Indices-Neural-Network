from pathlib import Path


# Data
# ------------------------------------------------
TRAIN_RGB_PATH = Path("../Jim/dataset/20meter/train_20meter_RGB_f32.npy")
TRAIN_NDVI_PATH = Path("../Jim/dataset/20meter/train_20meter_NDVI_f32.npy")
TEST_RGB_PATH = Path("../Jim/dataset/testing/test_15meter_RGB_f32.npy")
TEST_NDVI_PATH = Path("../Jim/dataset/testing/test_15meter_NDVI_f32.npy")
OUTPUT_DEFAULT_PATH = Path("outputs/default")
SAVE_IMAGE_PATH = OUTPUT_DEFAULT_PATH.joinpath("image")
SAVE_FIGURE_PATH = OUTPUT_DEFAULT_PATH.joinpath("figure")
RESAMPLE_MULTIPLE_FACTOR = 9


# Model
# ------------------------------------------------
MODEL_NAME = 'model'
SAVE_MODEL_PATH = OUTPUT_DEFAULT_PATH.joinpath("model")
SAVE_WEIGHT_PATH = OUTPUT_DEFAULT_PATH.joinpath("model")
TRAIN_INPUT_DIM = (352, 480, 3)
L2_REGULAR = 0.01


# Preprocessing
# ------------------------------------------------
ENABLE_DATA_AUG = False
USE_IMAGEDATAGENERATOR = False
SEED = 1


# Train
# ------------------------------------------------
EPOCHS = 200
TRAIN_BATCH_SIZE = 32
VAL_SPLIT = 0.1
EARLY_STOP_LOSS = 0.02
LEARNING_RATE_ARGS = dict(
    initial_learning_rate=0.001,
    decay_steps=1000,
    decay_rate=0.98,
    staircase=False
)


# ImageDataGenerator
# ------------------------------------------------
DATAGEN_ARGS = dict(
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=20,
    # width_shift_range=0.1,
    # height_shift_range=0.1,
    brightness_range=(0.9, 1.1),
    # shear_range=0.3,
    zoom_range=0.3,
    channel_shift_range=0.1,
    # rescale=1/255.,
    # featurewise_center=False,
    # samplewise_center=False,
    # featurewise_std_normalization=False,
    # samplewise_std_normalization=False,
    # zca_whitening=False,
    # zca_epsilon=1e-06,
    fill_mode='reflect',
    # cval=0.0,
    # preprocessing_function=None,
    # data_format=None,
    validation_split=VAL_SPLIT,
    # dtype=None,
)


# TensorBoard
# ------------------------------------------------
TENSORBOARD_LOG_PATH = OUTPUT_DEFAULT_PATH.joinpath("TensorBoard")
TENSORBOARD_ARGS = dict(
    log_dir=TENSORBOARD_LOG_PATH,
    histogram_freq=1,
    write_graph=False,
    write_images=True,
    update_freq='epoch',
    profile_batch=2,
    embeddings_freq=0,
    embeddings_metadata=None
)
