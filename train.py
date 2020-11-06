import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import optimizers
from keras.callbacks import Callback, EarlyStopping
from keras.initializers import glorot_normal, he_normal
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator

from model import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
data_used_amount = 246


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
            warnings.warn("Early stopping requires %s available!" %
                          self.monitor, RuntimeWarning)

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
        ax.imshow(images2[idx+i], vmin=-1, vmax=1)
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


def data_preprocessing():
    rgb_path = os.path.join('..', 'Jim', 'dataset','20meter', 'train_20meter_RGB.npy')
    ndvi_path = os.path.join('..', 'Jim', 'dataset','20meter', 'train_20meter_NDVI.npy')
    rgb_image_array = np.load(rgb_path, allow_pickle=True)
    ndvi_image_array = np.load(ndvi_path, allow_pickle=True)
    train_X = rgb_image_array.astype('float32') / 255.
    train_Y = ndvi_image_array.astype('float32')
    train_Y = np.expand_dims(train_Y, axis=3)
    return train_X, train_Y


if __name__ == "__main__":
    train_X, train_Y = data_preprocessing()

    print('RGB  array shape: ', train_X.shape)
    print('NDVI array shape: ', train_Y.shape)

    plot_multiimages(train_X, train_Y, 'RGB and NDVI Images', 72, 16)

    """
    datagen = ImageDataGenerator(
        zca_whitening=False,
        horizontal_flip=True,
        vertical_flip=True,
        #brightness_range=(0.1,0.5,1.0),
        #rotation_range=180,
        #width_shift_range=0.3,
        #height_shift_range=0.3,
    )
    """

    Model = AEN_model_1()
    adam = optimizers.Adam(lr=0.001)
    callbacks = [EarlyStoppingByLossVal(monitor='loss', value=1e-7, verbose=1)]
    Model.compile(optimizer=adam, loss='mean_squared_error')
    Model.summary()

    train_history = Model.fit(train_X[:data_used_amount], train_Y[:data_used_amount], epochs=20, batch_size=1, callbacks=callbacks, validation_split=0.1)

    """
    train_history = Model.fit_generator(datagen.flow(train_X[:], train_Y[:], batch_size=2, shuffle=True),
                                        epochs=80,
                                        samples_per_epoch=1000,
                                        verbose=2,
                                        callbacks=callbacks)
    """

    Model.save_weights('./weights/trained_model.h5')
    show_train_history(train_history, 'loss', 'val_loss')
