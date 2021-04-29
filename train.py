import os
import random as python_random
import sys

''' TF_CPP_MIN_LOG_LEVEL
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['PYTHONHASHSEED'] = '0'

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from cfgs import cfg
from models.unet_new import unet_new as Model
from utils.callback import SaveWeightCallback, TimingCallback
from utils.dataset import ImageDataSet
from utils.helper import output_init, plot_train_history, print_cfg
from utils.image import plot_two_images_array
from utils.preprocessing import train_preprocessing

np.random.seed(cfg.SEED)
python_random.seed(cfg.SEED)
tf.random.set_seed(cfg.SEED)

os.system('nvcc -V')
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


if __name__ == "__main__":
    fit_verbose = 1
    steps_per_execution = 1

    if len(sys.argv) > 1 and sys.argv[1] == "--prod":
        fit_verbose = 2
        steps_per_execution = 10000

    print_cfg(cfg)
    output_init(cfg)

    train_X_obj = ImageDataSet("RGB ", input_path=cfg.TRAIN_RGB_PATH, save_image_path=cfg.SAVE_IMAGE_PATH.joinpath("train/input"))
    train_Y_obj = ImageDataSet("NDVI", input_path=cfg.TRAIN_NDVI_PATH, save_image_path=cfg.SAVE_IMAGE_PATH.joinpath("train/input"))
    train_X_obj = train_X_obj.load_data(devided_by_255=False, expand_dims=False).crop()
    train_Y_obj = train_Y_obj.load_data(devided_by_255=False, expand_dims=False).crop()
    table = ImageDataSet.generate_resample_table(train_X_obj.num, cfg.TRAIN_RESAMPLE_MULTIPLE_FACTOR, (train_X_obj.height, train_X_obj.width), cfg.TRAIN_RESAMPLE_DIM)
    train_X = train_X_obj.resample(table, cfg.TRAIN_RESAMPLE_DIM)
    train_Y = train_Y_obj.resample(table, cfg.TRAIN_RESAMPLE_DIM)
    train_X = train_X_obj.downscale(cfg.TRAIN_DOWNSCALE_FACTOR)
    train_Y = train_Y_obj.downscale(cfg.TRAIN_DOWNSCALE_FACTOR)
    train_X = train_X_obj.get_image_array()
    train_Y = train_Y_obj.get_image_array()
    print('RGB  array shape: ', train_X.shape)
    print('NDVI array shape: ', train_Y.shape)

    plot_two_images_array(train_X, train_Y, 'Train - RGB, NDVI', cfg.SAVE_FIGURE_PATH)

    batch_size = cfg.TRAIN_BATCH_SIZE
    val_split = cfg.VAL_SPLIT
    steps_per_epoch = int(np.round(train_X.shape[0] / batch_size * (1 - val_split)))
    validation_steps = int(np.round(train_X.shape[0] / batch_size * val_split))

    lr_schedule = ExponentialDecay(**cfg.LEARNING_RATE_ARGS)

    early_stop_callback = EarlyStopping(**cfg.EARLY_STOP_ARGS)
    save_weight_callback = SaveWeightCallback(save_weight_path=cfg.SAVE_WEIGHT_PATH)
    timing_callback = TimingCallback(epochs=cfg.EPOCHS)
    tensorboard_callback = TensorBoard(**cfg.TENSORBOARD_ARGS)
    callbacks = [early_stop_callback, save_weight_callback, timing_callback, tensorboard_callback]

    model = Model(model_name=cfg.MODEL_NAME, input_dim=train_X.shape[1:])
    adam = Adam(learning_rate=lr_schedule)
    model.compile(optimizer=adam, loss='mean_absolute_error', metrics=RootMeanSquaredError(), steps_per_execution=steps_per_execution)
    model.summary()

    train_ds, validation_ds = train_preprocessing(train_X, train_Y, batch_size=batch_size, cfg=cfg)

    # dataset_plot_batch(train_ds, 10, "train", cfg.SAVE_FIGURE_PATH)
    # dataset_plot_batch(validation_ds, 10, "val", cfg.SAVE_FIGURE_PATH)

    train_history = model.fit(
        train_ds,
        epochs=cfg.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_ds,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=fit_verbose,
    )

    model.save(cfg.SAVE_MODEL_PATH.joinpath("trained_model.h5"))
    plot_train_history(train_history, 'loss', 'val_loss', save_figure_path=cfg.SAVE_FIGURE_PATH)

    print("Average epoch time: {0:.2f}s".format(np.mean(timing_callback.times)))
