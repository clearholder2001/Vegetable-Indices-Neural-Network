'''
tf.data version #2
-----------------------------------------------------
keras preprocessing layer
'''

import os
import sys
import warnings
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
from tensorflow.keras import optimizers, Sequential
from tensorflow.keras.callbacks import Callback, TensorBoard
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers.experimental import preprocessing

from configs import cfgs
from dataset import *
from model import *

gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

AUTOTUNE = tf.data.AUTOTUNE


# 設定迭代停止器
# 當loss function 低於某個值時，迭代自動停止
class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_loss', value=0.0001, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True
        # save the weights in every epoch
        self.model.save_weights("./weights/VGG_%d.h5" % epoch)


class TimingCallback(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time() - self.epoch_time_start)


def show_train_history(train_history, train, validation):
    fig = plt.figure(figsize=(8, 6))
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title = "Train History"
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('./fig/Train History.png')
    plt.close(fig)


def prepare(ds, batch_size, data_augmentation_image, data_augmentation_mask):
    ds = ds.batch(batch_size)
    ds = ds.map(lambda x, y: (data_augmentation_image(x, training=True), data_augmentation_mask(y, training=True)), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(buffer_size=AUTOTUNE)


'''
def image_preprocessing(image):
    assert image.ndim==3, 'image ndim != 3'
    if image.shape[-1] == (3 or 4):
        # process for color image
        pass
    return image
'''


if __name__ == "__main__":
    fit_verbose = 1

    if sys.argv[1] == "--production":
        fit_verbose = 2

    train_X_obj = DataObject('RGB ', cfgs.TRAIN_RGB_PATH)
    train_Y_obj = DataObject('NDVI', cfgs.TRAIN_NDVI_PATH)
    train_X_obj.load_data(devided_by_255=True, expand_dims=False, save_image=False)
    train_Y_obj.load_data(devided_by_255=False, expand_dims=True, save_image=False)
    train_X_obj.crop(save_image=False)
    train_Y_obj.crop(save_image=False)
    table = train_X_obj.generate_resample_table(multiple_factor=cfgs.RESAMPLE_MULTIPLE_FACTOR)
    train_X_obj.resample(table, save_image=False)
    train_Y_obj.resample(table, save_image=False)
    train_X = train_X_obj.get_data_resample()
    train_Y = train_Y_obj.get_data_resample()
    train_X, train_Y = shuffle(train_X, train_Y)

    val_split_count = int(train_X.shape[0] * (1 - cfgs.VAL_SPLIT))
    train_ds = tf.data.Dataset.from_tensor_slices((train_X, train_Y)).take(val_split_count)
    val_ds = tf.data.Dataset.from_tensor_slices((train_X, train_Y)).skip(val_split_count)

    plot_two_images_array(train_X, train_Y, 'Train - RGB, NDVI', 0)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=cfgs.INIT_LEARNING_RATE,
        decay_steps=cfgs.DECAY_STEPS,
        decay_rate=cfgs.DECAY_RATE,
        staircase=cfgs.STAIRCASE
    )

    Model = AE_model_4(cfgs.MODEL_NAME)
    adam = optimizers.Adam(learning_rate=lr_schedule)
    Model.compile(optimizer=adam, loss='mean_absolute_error')
    Model.summary()

    early_stop_callback = EarlyStoppingByLossVal(monitor='loss', value=1e-3, verbose=1)
    timing_callback = TimingCallback()
    tensorboard_callback = TensorBoard(
        log_dir='tb_log',
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq='epoch',
        profile_batch=2,
        embeddings_freq=1,
        embeddings_metadata=None
    )
    callbacks = [early_stop_callback, timing_callback, tensorboard_callback]

    data_used_amount = train_X.shape[0]
    seed = int(time())
    batch_size = cfgs.DATA_AUG_BATCH_SIZE
    steps_per_epoch = int(np.ceil((data_used_amount / batch_size) * (1 - cfgs.VAL_SPLIT)))
    validation_steps = int(np.ceil((data_used_amount / batch_size) * cfgs.VAL_SPLIT))

    data_augmentation_image = tf.keras.Sequential([
        preprocessing.RandomFlip(mode="horizontal_and_vertical", seed=seed),
        preprocessing.RandomRotation(factor=np.radians(cfgs.DATAGEN_ARGS["rotation_range"]), fill_mode=cfgs.DATAGEN_ARGS["fill_mode"], interpolation="bilinear", seed=seed),
        preprocessing.RandomZoom(height_factor=cfgs.DATAGEN_ARGS["zoom_range"], width_factor=cfgs.DATAGEN_ARGS["zoom_range"], fill_mode=cfgs.DATAGEN_ARGS["fill_mode"], interpolation="bilinear", seed=seed),
        preprocessing.RandomContrast(factor=cfgs.RANDOMCONTRAST_FACTOR, seed=seed),
    ])
    data_augmentation_mask = tf.keras.Sequential([
        preprocessing.RandomFlip(mode="horizontal_and_vertical", seed=seed),
        preprocessing.RandomRotation(factor=np.radians(cfgs.DATAGEN_ARGS["rotation_range"]), fill_mode=cfgs.DATAGEN_ARGS["fill_mode"], interpolation="bilinear", seed=seed),
        preprocessing.RandomZoom(height_factor=cfgs.DATAGEN_ARGS["zoom_range"], width_factor=cfgs.DATAGEN_ARGS["zoom_range"], fill_mode=cfgs.DATAGEN_ARGS["fill_mode"], interpolation="bilinear", seed=seed),
    ])

    train_ds = prepare(train_ds, batch_size, data_augmentation_image, data_augmentation_mask)
    val_ds = prepare(val_ds, batch_size, data_augmentation_image, data_augmentation_mask)

    if cfgs.ENABLE_DATA_AUG:
        train_history = Model.fit(
            train_ds,
            epochs=cfgs.EPOCHS,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_ds,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=fit_verbose
        )
    else:
        train_history = Model.fit(
            train_X[:data_used_amount],
            train_Y[:data_used_amount],
            epochs=cfgs.EPOCHS,
            steps_per_epoch=None,
            batch_size=cfgs.TRAIN_BATCH_SIZE,
            shuffle=True,
            validation_split=cfgs.VAL_SPLIT,
            callbacks=callbacks,
            verbose=fit_verbose
        )

    Model.save_weights('./weights/trained_model.h5')
    show_train_history(train_history, 'loss', 'val_loss')

    print("Average epoch time: {0}s".format(str(np.mean(timing_callback.times))))
