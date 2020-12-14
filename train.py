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
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from configs import cfgs
from dataset import *
from model import *

gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


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


def plot_multiimages(images1, images2, title, idx, num=16):
    plt.gcf().set_size_inches(8, 6)
    if num > 16:
        num = 16
    for i in range(0, int(num/2)):
        ax = plt.subplot(4, 4, 1+i)
        ax.imshow(images1[idx+i], vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
    for i in range(0, int(num/2)):
        ax = plt.subplot(4, 4, int(num/2)+1+i)
        ax.imshow(images2[idx+i], vmin=-1, vmax=1, cmap=plt.get_cmap('jet'))
        ax.set_xticks([])
        ax.set_yticks([])
    plt.suptitle(title)
    plt.tight_layout()
    # plt.show()
    plt.savefig('./fig/' + title + '.png')
    plt.close()


def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title = "Train History"
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    # plt.show()
    plt.savefig('./fig/Train History.png')
    plt.close()


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
    train_X_obj.load_data(devided_by_255=True, expand_dims=False)
    train_Y_obj.load_data(devided_by_255=False, expand_dims=True)
    train_X_obj.crop()
    train_Y_obj.crop()
    table = train_X_obj.generate_resample_table(multiple_factor=cfgs.RESAMPLE_MULTIPLE_FACTOR)
    train_X_obj.resample(table)
    train_Y_obj.resample(table)
    train_X = train_X_obj.get_data_resample()
    train_Y = train_Y_obj.get_data_resample()
    train_X, train_Y = shuffle(train_X, train_Y)

    plot_multiimages(train_X, train_Y, 'RGB and NDVI Images', 0, 16)

    Model = AE_model_2(cfgs.MODEL_NAME)
    adam = optimizers.Adam(cfgs.LEARNING_RATE)
    callbacks = [EarlyStoppingByLossVal(monitor='loss', value=1e-3, verbose=1)]
    Model.compile(optimizer=adam, loss='mean_squared_error')
    Model.summary()

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

    train_generator = zip(train_image_generator, train_mask_generator)
    validation_generator = zip(validation_image_generator, validation_mask_generator)

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
