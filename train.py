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
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from cfgs import cfg
from models.model import AE_model_4
from utils.callback import EarlyStoppingByLossVal, TimingCallback
from utils.dataset import DataObject
from utils.helper import show_train_history
from utils.image import plot_two_images_array

gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

AUTOTUNE = tf.data.AUTOTUNE


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

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=cfg.INIT_LEARNING_RATE,
        decay_steps=cfg.DECAY_STEPS,
        decay_rate=cfg.DECAY_RATE,
        staircase=cfg.STAIRCASE
    )

    Model = AE_model_4(cfg.MODEL_NAME)
    adam = optimizers.Adam(learning_rate=lr_schedule)
    Model.compile(optimizer=adam, loss='mean_absolute_error')
    Model.summary()

    early_stop_callback = EarlyStoppingByLossVal(monitor='loss', value=1e-3, verbose=1)
    timing_callback = TimingCallback()
    tensorboard_callback = TensorBoard(
        log_dir='tb_log',
        histogram_freq=0,
        write_graph=True,
        write_images=False,
        update_freq='epoch',
        profile_batch=2,
        embeddings_freq=0,
        embeddings_metadata=None
    )
    callbacks = [early_stop_callback, timing_callback, tensorboard_callback]

    data_used_amount = train_X.shape[0]
    seed = int(time())
    batch_size = cfg.DATA_AUG_BATCH_SIZE
    steps_per_epoch = int(np.ceil((data_used_amount / batch_size) * (1 - cfg.VAL_SPLIT)))
    validation_steps = int(np.ceil((data_used_amount / batch_size) * cfg.VAL_SPLIT))

    image_datagen = ImageDataGenerator(**cfg.DATAGEN_ARGS)
    mask_datagen = ImageDataGenerator(**cfg.DATAGEN_ARGS)
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

    train_generator = zip(train_image_generator, train_mask_generator)
    validation_generator = zip(validation_image_generator, validation_mask_generator)

    if cfg.ENABLE_DATA_AUG:
        train_history = Model.fit(
            train_generator,
            epochs=cfg.EPOCHS,
            steps_per_epoch=steps_per_epoch,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=2
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
            verbose=1
        )

    Model.save_weights('./weights/trained_model.h5')
    show_train_history(train_history, 'loss', 'val_loss')

    print("Average epoch time: {0:.2f}s".format(np.mean(timing_callback.times)))
