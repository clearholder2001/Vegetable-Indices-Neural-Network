import math
import os

''' TF_CPP_MIN_LOG_LEVEL
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib.image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy import stats
from sklearn.metrics import r2_score
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, load_model

from configs import cfgs
from dataset import *
from model import *

gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


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


def plot_multi_result(images1, images2, images3, title, idx):
    plt.gcf().set_size_inches(12, 13)
    for i in range(0, 4):
        ax = plt.subplot(4, 3, i*3+1)
        ax.imshow(images1[idx+i], vmin=0, vmax=1)
        ax = plt.subplot(4, 3, i*3+2)
        ax.imshow(images2[idx+i], vmin=-1, vmax=1, cmap=plt.get_cmap('jet'))
        ax = plt.subplot(4, 3, i*3+3)
        ax.imshow(images3[idx+i], vmin=-1, vmax=1, cmap=plt.get_cmap('jet'))
    plt.suptitle(title)
    plt.tight_layout()
    # plt.show()
    plt.savefig('./fig/' + title + '.png')
    plt.close()


def save_result_image(test_X, test_Y, predict, output_compare=True):
    assert test_X.shape[0] == test_Y.shape[0] == predict.shape[0], 'test_X, test_Y, preditc長度不一致'
    path = './fig/inference/'

    if output_compare:
        fig, axs = plt.subplots(1, 3)
        fig.set_size_inches(12, 4)
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
        axs[0].set_title("RGB")
        img1 = axs[0].imshow(test_X[0], vmin=0, vmax=1)
        axs[1].set_title("NDVI")
        img2 = axs[1].imshow(test_Y[0], vmin=-1, vmax=1, cmap=plt.get_cmap('jet'))
        axs[2].set_title("Predict")
        img3 = axs[2].imshow(predict[0], vmin=-1, vmax=1, cmap=plt.get_cmap('jet'))

    for i in range(test_X.shape[0]):
        matplotlib.image.imsave(path + 'rgb/rgb_{0}.jpg'.format(i), test_X[i])
        matplotlib.image.imsave(path + 'ndvi/ndvi_{0}.jpg'.format(i), np.squeeze(test_Y[i]), cmap=plt.get_cmap('jet'))
        matplotlib.image.imsave(path + 'predict/predict_{0}.jpg'.format(i), np.squeeze(predict[i]), cmap=plt.get_cmap('jet'))
        if output_compare:
            img1.set_data(test_X[i])
            img2.set_data(test_Y[i])
            img3.set_data(predict[i])
            fig.suptitle("Compare #" + str(i), fontsize=24)
            fig.tight_layout()
            fig.savefig(path + 'compare/compare_{0}.jpg'.format(i))
    plt.close()


if __name__ == "__main__":
    cfgs.RESAMPLE_MULTIPLE_FACTOR = 1
    weight_name = 'trained_model.h5'

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

    Model = AE_model_3(cfgs.MODEL_NAME)
    adam = optimizers.Adam(cfgs.INIT_LEARNING_RATE)
    Model.compile(optimizer=adam, loss='mean_absolute_error')
    weight = os.path.join('.', 'weights', weight_name)
    Model.load_weights(weight)

    batch_size = cfgs.TRAIN_BATCH_SIZE
    predict = Model.predict(test_X, batch_size=batch_size, verbose=2)
    lossfunc = Model.evaluate(test_X, test_Y, batch_size=batch_size, verbose=2)
    assert predict.shape == test_Y.shape, 'Prediction維度和NDVI不同'

    #np.save('predict', predict, allow_pickle=True)
    plot_multi_result(test_X, test_Y, predict, 'Test - Result', 0)
    save_result_image(test_X, test_Y, predict, output_compare=True)

    num = test_Y.shape[0]
    rmse = math.sqrt(np.mean(np.square(test_Y - predict)))
    test_Y = np.reshape(test_Y, (num, -1))
    predict = np.reshape(predict, (num, -1))
    r2 = np.zeros(num)
    correlation = np.zeros((num, 2))
    for i in range(num):
        r2[i] = r2_score(test_Y[i], predict[i])
        correlation[i] = stats.pearsonr(test_Y[i], predict[i])
    print('Final RMSE: ', rmse)
    print('Final R Square: ', np.mean(r2))
    print('Final Correlation: ', np.mean(correlation[:, 0]))
    print("Final Loss: ", lossfunc)
