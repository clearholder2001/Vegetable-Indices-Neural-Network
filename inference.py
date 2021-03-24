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
from tensorflow.keras.models import Model, load_model

from cfgs import cfg
from utils.dataset import DataObject
from utils.image import (plot_three_images_array, plot_two_images_array,
                         save_result_image)

gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


if __name__ == "__main__":
    model_path = cfg.SAVE_MODEL_PATH.joinpath("trained_model.h5")

    test_X_obj = DataObject('RGB ', cfg.TEST_RGB_PATH, save_image_path=cfg.SAVE_IMAGE_PATH.joinpath("inference/input"))
    test_Y_obj = DataObject('NDVI', cfg.TEST_NDVI_PATH, save_image_path=cfg.SAVE_IMAGE_PATH.joinpath("inference/input"))
    test_X_obj.load_data(devided_by_255=True, expand_dims=False, save_image=False)
    test_Y_obj.load_data(devided_by_255=False, expand_dims=True, save_image=False)
    test_X_obj.crop(save_image=False)
    test_Y_obj.crop(save_image=False)
    table = test_X_obj.generate_resample_table(multiple_factor=cfg.RESAMPLE_MULTIPLE_FACTOR, seed=cfg.SEED)
    test_X_obj.resample(table, save_image=False)
    test_Y_obj.resample(table, save_image=False)
    test_X = test_X_obj.get_data_resample()
    test_Y = test_Y_obj.get_data_resample()
    print('RGB  array shape: ', test_X.shape)
    print('NDVI array shape: ', test_Y.shape)

    plot_two_images_array(test_X, test_Y, 'Inference - RGB, NDVI', 0, cfg.SAVE_FIGURE_PATH)

    model = Model(model_name=cfg.MODEL_NAME, input_dim=cfg.INFERENCE_INPUT_DIM)
    model.compile(loss='mean_absolute_error')
    model.load_weights(model_path)
    model.summary()

    batch_size = cfg.TRAIN_BATCH_SIZE
    predict = model.predict(test_X, batch_size=batch_size, verbose=2)
    lossfunc = model.evaluate(test_X, test_Y, batch_size=batch_size, verbose=2)
    assert predict.shape == test_Y.shape, 'Dimension inconsistent: test_Y, predict'

    num = test_Y.shape[0]
    rmse = math.sqrt(np.mean(np.square(test_Y - predict)))
    r2 = r2_score(test_Y.reshape(-1), predict.reshape(-1))
    correlation = stats.pearsonr(test_Y.reshape(-1), predict.reshape(-1))
    print("Final RMSE: {0:.4f}".format(rmse))
    print("Final R Square: {0:.4f}".format(r2))
    print("Final Correlation: {0:.4f}".format(correlation[0]))
    print("Final Loss ({0}): {1:.4f}".format(model.loss, lossfunc))

    np.save(cfg.OUTPUT_DEFAULT_PATH.joinpath("predict"), predict, allow_pickle=True)
    plot_three_images_array(test_X, test_Y, predict, 'Inference - RGB, NDVI, Predict', 0, cfg.SAVE_FIGURE_PATH)
    save_result_image(test_X, test_Y, predict, output_compare=True, save_image_path=cfg.SAVE_IMAGE_PATH.joinpath("inference/output"))
