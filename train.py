import os
import sys
from time import time

''' TF_CPP_MIN_LOG_LEVEL
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import Callback, TensorBoard
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from cfgs import cfg
from models import unet_C2DT as model
from utils.callback import EarlyStoppingByLossVal, TimingCallback
from utils.data_aug import data_aug_layer_tf_dataset as data_augmentation
from utils.dataset import DataObject
from utils.helper import show_train_history
from utils.image import plot_two_images_array

gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


if __name__ == "__main__":
    fit_verbose = 1

    if len(sys.argv) > 1 and sys.argv[1] == "--production":
        fit_verbose = 2

    train_X_obj = DataObject('RGB ', cfg.TRAIN_RGB_PATH)
    train_Y_obj = DataObject('NDVI', cfg.TRAIN_NDVI_PATH)
    train_X_obj.load_data(devided_by_255=True, expand_dims=False, save_image=False)
    train_Y_obj.load_data(devided_by_255=False, expand_dims=True, save_image=False)
    train_X_obj.crop(save_image=False)
    train_Y_obj.crop(save_image=False)
    table = train_X_obj.generate_resample_table(multiple_factor=cfg.RESAMPLE_MULTIPLE_FACTOR)
    train_X_obj.resample(table, save_image=False)
    train_Y_obj.resample(table, save_image=False)
    train_X = train_X_obj.get_data_resample()
    train_Y = train_Y_obj.get_data_resample()
    train_X, train_Y = shuffle(train_X, train_Y)

    plot_two_images_array(train_X, train_Y, 'Train - RGB, NDVI', 0)

    train_ds, validation_ds = data_augmentation(train_X, train_Y)

    lr_schedule = ExponentialDecay(**cfg.LEARNING_RATE_ARGS)

    early_stop_callback = EarlyStoppingByLossVal(monitor='loss', value=1e-3, verbose=1)
    timing_callback = TimingCallback()
    tensorboard_callback = TensorBoard(**cfg.TENSORBOARD_ARGS)
    callbacks = [early_stop_callback, timing_callback, tensorboard_callback]

    data_used_amount = train_X.shape[0]
    batch_size = cfg.DATA_AUG_BATCH_SIZE
    split_ratio = cfg.VAL_SPLIT
    steps_per_epoch = int(np.ceil((data_used_amount / batch_size) * (1 - split_ratio)))
    validation_steps = int(np.ceil((data_used_amount / batch_size) * split_ratio))

    Model = model(cfg.MODEL_NAME)
    adam = optimizers.Adam(learning_rate=lr_schedule)
    Model.compile(optimizer=adam, loss='mean_absolute_error')
    Model.summary()

    if cfg.ENABLE_DATA_AUG:
        train_history = Model.fit(
            train_ds,
            epochs=cfg.EPOCHS,
            steps_per_epoch=steps_per_epoch,
            validation_data=validation_ds,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=fit_verbose
        )
    else:
        train_history = Model.fit(
            train_X[:data_used_amount],
            train_Y[:data_used_amount],
            epochs=cfg.EPOCHS,
            steps_per_epoch=None,
            batch_size=cfg.TRAIN_BATCH_SIZE,
            shuffle=True,
            validation_split=cfg.VAL_SPLIT,
            callbacks=callbacks,
            verbose=fit_verbose
        )

    Model.save_weights('./weights/trained_model.h5')
    show_train_history(train_history, 'loss', 'val_loss')

    print("Average epoch time: {0:.2f}s".format(np.mean(timing_callback.times)))
