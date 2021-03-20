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

from cfgs import cfg
from models import unet_C2DT as model
from utils.dataset import DataObject
from utils.image import (plot_three_images_array, plot_two_images_array,
                         save_result_image)

gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


if __name__ == "__main__":
    cfg.RESAMPLE_MULTIPLE_FACTOR = 1
    weight_name = 'trained_model.h5'

    test_X_obj = DataObject('RGB ', cfg.TEST_RGB_PATH)
    test_Y_obj = DataObject('NDVI', cfg.TEST_NDVI_PATH)
    test_X_obj.load_data(devided_by_255=True, expand_dims=False)
    test_Y_obj.load_data(devided_by_255=False, expand_dims=True)
    test_X_obj.crop()
    test_Y_obj.crop()
    table = test_X_obj.generate_resample_table(multiple_factor=cfg.RESAMPLE_MULTIPLE_FACTOR)
    test_X_obj.resample(table)
    test_Y_obj.resample(table)
    test_X = test_X_obj.get_data_resample()
    test_Y = test_Y_obj.get_data_resample()
    print('RGB  array shape: ', test_X.shape)
    print('NDVI array shape: ', test_Y.shape)

    plot_two_images_array(test_X, test_Y, 'Inference - RGB, NDVI', 140)

    # cfg.INPUT_LAYER_DIM = (test_X.shape[1], test_X.shape[2], test_X.shape[3])

    Model = model(cfg.MODEL_NAME)
    adam = optimizers.Adam(cfg.INIT_LEARNING_RATE)
    Model.compile(optimizer=adam, loss='mean_absolute_error')
    weight = os.path.join('.', 'weights', weight_name)
    Model.load_weights(weight)

    batch_size = cfg.TRAIN_BATCH_SIZE
    predict = Model.predict(test_X, batch_size=batch_size, verbose=2)
    lossfunc = Model.evaluate(test_X, test_Y, batch_size=batch_size, verbose=2)
    assert predict.shape == test_Y.shape, 'Dimension inconsistent: test_Y, predict'

    #np.save('predict', predict, allow_pickle=True)
    plot_three_images_array(test_X, test_Y, predict, 'Inference - RGB, NDVI, Predict', 0)
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
