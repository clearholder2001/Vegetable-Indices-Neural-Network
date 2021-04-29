from pathlib import Path


# Path
# ------------------------------------------------
TRAIN_RGB_PATH = Path("../Jim/dataset/20meter/train_20meter_RGB_f32.npy")
TRAIN_NDVI_PATH = Path("../Jim/dataset/20meter/train_20meter_NDVI_f32.npy")
TEST_RGB_PATH = Path("../Jim/dataset/testing/test_15meter_RGB_f32.npy")
TEST_NDVI_PATH = Path("../Jim/dataset/testing/test_15meter_NDVI_f32.npy")
OUTPUT_DEFAULT_PATH = Path("outputs/default")
SAVE_IMAGE_PATH = OUTPUT_DEFAULT_PATH.joinpath("image")
SAVE_FIGURE_PATH = OUTPUT_DEFAULT_PATH.joinpath("figure")
SAVE_MODEL_PATH = OUTPUT_DEFAULT_PATH.joinpath("model")
SAVE_WEIGHT_PATH = OUTPUT_DEFAULT_PATH.joinpath("model")
TENSORBOARD_LOG_PATH = OUTPUT_DEFAULT_PATH.joinpath("TensorBoard")


# Model
# ------------------------------------------------
MODEL_NAME = 'model'
TRAIN_INPUT_DIM = (224, 304, 3)
TEST_INPUT_DIM = (224, 304, 3)
L2_REGULAR = 0.01
SEED = 1


# Preprocessing
# ------------------------------------------------
TRAIN_RESCALE_FACTOR = 0.25
TEST_RESCALE_FACTOR = 0.25
TRAIN_RESAMPLE_FACTOR = 1
TEST_RESAMPLE_FACTOR = 1
ENABLE_DATA_AUG = False
USE_IMAGEDATAGENERATOR = False


# Train
# ------------------------------------------------
EPOCHS = 200
BATCH_SIZE = 16
VAL_SPLIT = 0.1
LEARNING_RATE_ARGS = dict(
    initial_learning_rate=0.001,
    decay_steps=100,
    decay_rate=0.98,
    staircase=False
)
EARLY_STOP_ARGS = dict(
    monitor='loss',
    min_delta=0.0005,
    patience=20,
    verbose=1,
    mode='min',
    baseline=0.2
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
TENSORBOARD_ARGS = dict(
    log_dir=TENSORBOARD_LOG_PATH,
    histogram_freq=1,
    write_graph=True,
    write_images=False,
    update_freq='epoch',
    profile_batch=2,
    embeddings_freq=0,
    embeddings_metadata=None
)
