import math
import os

''' TF_CPP_MIN_LOG_LEVEL
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as state
import tensorflow as tf
from sklearn.metrics import r2_score
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, load_model

from configs import cfgs
from dataset import *
from model import *


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


def plot_single_result(images1, images2, images3, title, idx):
    plt.gcf().set_size_inches(9, 4)
    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(images1[idx], vmin=0, vmax=1)
    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(images2[idx], vmin=-1, vmax=1, cmap=plt.get_cmap('jet'))
    ax3 = plt.subplot(1, 3, 3)
    ax3.imshow(images3[idx], vmin=-1, vmax=1, cmap=plt.get_cmap('jet'))
    plt.suptitle(title)
    plt.axis('off')
    plt.tight_layout()
    # plt.show()
    plt.savefig('./fig/' + title + '.png')
    plt.close()


if __name__ == "__main__":
    cfgs.RESAMPLE_MULTIPLE_FACTOR = 1

    test_X_obj = DataObject('RGB ', cfgs.TEST_RGB_PATH)
    test_Y_obj = DataObject('NDVI', cfgs.TEST_NDVI_PATH)
    test_X_obj.load_data(devided_by_255=True, expand_dims=False)
    test_Y_obj.load_data(devided_by_255=False, expand_dims=True)
    test_X_obj.crop()
    test_Y_obj.crop()
    table = test_X_obj.generate_resample_table(multiple_factor=cfgs.RESAMPLE_MULTIPLE_FACTOR)
    test_X_obj.resample(table)
    test_Y_obj.resample(table)
    test_X = test_X_obj.get_data_resample()
    test_Y = test_Y_obj.get_data_resample()
    print('RGB  array shape: ', test_X.shape)
    print('NDVI array shape: ', test_Y.shape)

    plot_multiimages(test_X, test_Y, 'Test - RGB and NDVI Images', 140, 16)

    # cfgs.INPUT_LAYER_DIM = (test_X.shape[1], test_X.shape[2], test_X.shape[3])

    Model = AE_model_2()
    adam = optimizers.Adam(cfgs.LEARNING_RATE)
    Model.compile(optimizer=adam, loss='mean_squared_error')
    weight = os.path.join('.', 'weights', 'trained_model.h5')
    Model.load_weights(weight)

    predict = Model.predict(test_X)
    lossfunc = Model.evaluate(test_X, test_Y)
    assert predict.shape == test_Y.shape, 'Prediction維度和NDVI不同'
    
    rmse = math.sqrt(np.mean(np.square(test_Y - predict)))
    # r2 = r2_score(test_Y, predict)
    #correlation = state.pearsonr(test_Y, predict)
    print('Final RMSE: ', rmse)
    # print('Final R Square: ', r2)
    #print('Final Correlation: ', correlation[0])
    print("Final Loss: ", lossfunc)

    #np.save('predict', predict, allow_pickle=True)
    plot_single_result(test_X, test_Y, predict, 'Test - Result', 140)

