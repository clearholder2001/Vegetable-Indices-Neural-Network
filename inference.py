import os
import random as python_random
import sys

''' TF_CPP_MIN_LOG_LEVEL
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['PYTHONHASHSEED'] = '0'

import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import RootMeanSquaredError

from cfgs import cfg
from models.unet_new import unet_new as Model
from utils.dataset import ImageDataSet
from utils.helper import calculate_statistics
from utils.image import (plot_three_images_array, plot_two_images_array,
                         save_result_image)
from utils.preprocessing import test_precessing

np.random.seed(cfg.SEED)
python_random.seed(cfg.SEED)
tf.random.set_seed(cfg.SEED)

os.system('nvcc -V')
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


def predict_function(model, test_ds, predict_shape, steps, verbose):
    predict = np.zeros(predict_shape, dtype=np.float32)
    for idx, (image_batch, None) in enumerate(test_ds.as_numpy_iterator()):
        begin = idx * batch_size
        end = begin + image_batch.shape[0]
        predict[begin:end, :, :, :] = model.predict_on_batch(image_batch)
        if verbose and (idx+1) != steps:
            print(f"Predicting...{idx+1}/{steps}", end='\r')
        elif (idx+1) == steps:
            print(f"Predicting...{idx+1}/{steps}")
    return predict


if __name__ == "__main__":
    test_verbose = 1

    if len(sys.argv) > 1 and sys.argv[1] == "--prod":
        test_verbose = 0

    model_path = cfg.SAVE_MODEL_PATH.joinpath("trained_model.h5")

    test_X_obj = ImageDataSet("RGB ", input_path=cfg.TEST_RGB_PATH, save_image_path=cfg.SAVE_IMAGE_PATH.joinpath("inference/input"))
    test_Y_obj = ImageDataSet("NDVI", input_path=cfg.TEST_NDVI_PATH, save_image_path=cfg.SAVE_IMAGE_PATH.joinpath("inference/input"))
    test_X_obj = test_X_obj.load_data(devided_by_255=False, expand_dims=False).crop()
    test_Y_obj = test_Y_obj.load_data(devided_by_255=False, expand_dims=False).crop()
    table = ImageDataSet.generate_resample_table(test_X_obj.num, cfg.TEST_RESAMPLE_MULTIPLE_FACTOR, (test_X_obj.height, test_X_obj.width), cfg.TEST_RESAMPLE_DIM)
    test_X = test_X_obj.resample(table, cfg.TEST_RESAMPLE_DIM)
    test_Y = test_Y_obj.resample(table, cfg.TEST_RESAMPLE_DIM)
    test_X = test_X_obj.downscale(cfg.TEST_DOWNSCALE_FACTOR)
    test_Y = test_Y_obj.downscale(cfg.TEST_DOWNSCALE_FACTOR)
    test_X = test_X_obj.get_image_array()
    test_Y = test_Y_obj.get_image_array()
    print(f"RGB  array shape: {test_X.shape}")
    print(f"NDVI array shape: {test_Y.shape}")

    plot_two_images_array(test_X, test_Y, 'Inference - RGB, NDVI', cfg.SAVE_FIGURE_PATH)

    model = Model(model_name=cfg.MODEL_NAME, input_dim=test_X.shape[1:])
    model.compile(loss='mean_absolute_error', metrics=RootMeanSquaredError())
    model.load_weights(model_path)
    model.summary()

    batch_size = cfg.TEST_BATCH_SIZE
    steps = int(np.ceil(test_X.shape[0] / batch_size))
    test_ds = test_precessing(test_X, test_Y, batch_size)

    predict = predict_function(model, test_ds, test_Y.shape, steps, test_verbose)
    loss = model.evaluate(test_ds, verbose=test_verbose)
    calculate_statistics(test_Y, predict)

    np.save(cfg.OUTPUT_DEFAULT_PATH.joinpath("predict.npy"), predict, allow_pickle=True)
    plot_three_images_array(test_X, test_Y, predict, 'Inference - RGB, NDVI, Predict', cfg.SAVE_FIGURE_PATH)
    save_result_image(test_X, test_Y, predict, output_compare=True, save_image_path=cfg.SAVE_IMAGE_PATH.joinpath("inference/output"))
