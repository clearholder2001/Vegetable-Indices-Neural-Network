'''
tf.data version #1
-----------------------------------------------------
ImageDataGenerator + tf.data.Dataset.from_generator
'''

import os
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
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import Callback, TensorBoard
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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


'''
def image_preprocessing(image):
    assert image.ndim==3, 'image ndim != 3'
    if image.shape[-1] == (3 or 4):
        # process for color image
        pass
    return image
'''


if __name__ == "__main__":
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

    image_datagen = ImageDataGenerator(**cfgs.DATAGEN_ARGS)
    mask_datagen = ImageDataGenerator(**cfgs.DATAGEN_ARGS)
    #image_datagen.preprocessing_function = image_preprocessing

    train_image_generator = image_datagen.flow(
        train_X,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        subset='training',  # set as training data
        #save_to_dir='./fig/datagen/train/rgb',
        #save_prefix='train',
        #save_format='jpg'
    )

    train_mask_generator = mask_datagen.flow(
        train_Y,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        subset='training',  # set as training data
        #save_to_dir='./fig/datagen/train/ndvi',
        #save_prefix='train',
        #save_format='jpg'
    )

    validation_image_generator = image_datagen.flow(
        train_X,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        subset='validation',  # set as validation data
        #save_to_dir='./fig/datagen/val/rgb',
        #save_prefix='val',
        #save_format='jpg'
    )

    validation_mask_generator = mask_datagen.flow(
        train_Y,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        subset='validation',  # set as validation data
        #save_to_dir='./fig/datagen/val/ndvi',
        #save_prefix='val',
        #save_format='jpg'
    )

    train_image_generator_ds = tf.data.Dataset.from_generator(lambda: train_image_generator, output_types=tf.float32, output_shapes=[batch_size].extend(cfgs.INPUT_LAYER_DIM))
    train_mask_generator_ds = tf.data.Dataset.from_generator(lambda: train_mask_generator, output_types=tf.float32, output_shapes=[batch_size].extend(cfgs.INPUT_LAYER_DIM))
    validation_image_generator_ds = tf.data.Dataset.from_generator(lambda: validation_image_generator, output_types=tf.float32, output_shapes=[batch_size].extend(cfgs.INPUT_LAYER_DIM))
    validation_mask_generator_ds = tf.data.Dataset.from_generator(lambda: validation_mask_generator, output_types=tf.float32, output_shapes=[batch_size].extend(cfgs.INPUT_LAYER_DIM))

    train_image_generator_ds = train_image_generator_ds.prefetch(buffer_size=AUTOTUNE)
    train_mask_generator_ds = train_mask_generator_ds.prefetch(buffer_size=AUTOTUNE)
    validation_image_generator_ds = validation_image_generator_ds.prefetch(buffer_size=AUTOTUNE)
    validation_mask_generator_ds = validation_mask_generator_ds.prefetch(buffer_size=AUTOTUNE)

    train_generator = tf.data.Dataset.zip((train_image_generator_ds, train_mask_generator_ds))
    validation_generator = tf.data.Dataset.zip((validation_image_generator_ds, validation_mask_generator_ds))

    if cfgs.ENABLE_DATA_AUG:
        train_history = Model.fit(
            train_generator,
            epochs=cfgs.EPOCHS,
            steps_per_epoch=steps_per_epoch,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
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
            verbose=1
        )

    Model.save_weights('./weights/trained_model.h5')
    show_train_history(train_history, 'loss', 'val_loss')

    print("Average epoch time: {0}s".format(str(np.mean(timing_callback.times))))