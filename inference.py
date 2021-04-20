import os
import random as python_random

''' TF_CPP_MIN_LOG_LEVEL
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['PYTHONHASHSEED'] = '0'

import matplotlib.image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import config
from tensorflow.keras.metrics import RootMeanSquaredError

from cfgs import cfg
from models.unet_C2DT import unet_C2DT as Model
from utils.dataset import ImageDataSet
from utils.helper import (calculate_statistics, create_tf_dataset, output_init,
                          simple_image_generator)
from utils.image import (plot_three_images_array, plot_two_images_array,
                         save_result_image)

np.random.seed(cfg.SEED)
python_random.seed(cfg.SEED)
tf.random.set_seed(cfg.SEED)

gpus = config.list_physical_devices('GPU')
config.set_visible_devices(gpus[0], 'GPU')
config.experimental.set_memory_growth(gpus[0], True)


def predict_function(model, test_X, predict_shape, batch_size):
    predict = np.zeros((predict_shape))
    steps = int(np.ceil(test_X.shape[0] / batch_size))
    print("Predicting...", end='')
    for i in range(steps):
        begin = i * batch_size
        if i == steps - 1:
            end = test_X.shape[0]
        else:
            end = (i + 1) * batch_size
        predict[begin:end, :, :, :] = model.predict_on_batch(test_X[begin:end, :, :, :])
    print("Done")
    return predict


if __name__ == "__main__":
    model_path = cfg.SAVE_MODEL_PATH.joinpath("trained_model.h5")

    test_X_obj = ImageDataSet('RGB ', cfg.TEST_RGB_PATH, save_image_path=cfg.SAVE_IMAGE_PATH.joinpath("inference/input"))
    test_Y_obj = ImageDataSet('NDVI', cfg.TEST_NDVI_PATH, save_image_path=cfg.SAVE_IMAGE_PATH.joinpath("inference/input"))
    test_X_obj.load_data(devided_by_255=False, expand_dims=False, save_image=False)
    test_Y_obj.load_data(devided_by_255=False, expand_dims=False, save_image=False)
    test_X_obj.crop(save_image=False)
    test_Y_obj.crop(save_image=False)
    table = test_X_obj.generate_resample_table(multiple_factor=cfg.RESAMPLE_MULTIPLE_FACTOR, seed=cfg.SEED)
    test_X_obj.resample(table, save_image=False)
    test_Y_obj.resample(table, save_image=False)
    test_X = test_X_obj.get_data_resample()
    test_Y = test_Y_obj.get_data_resample()
    print('RGB  array shape: ', test_X.shape)
    print('NDVI array shape: ', test_Y.shape)

    batch_size = cfg.TRAIN_BATCH_SIZE
    test_X_ds = create_tf_dataset(test_X, simple_image_generator, batch_size=batch_size)
    test_Y_ds = create_tf_dataset(test_Y, simple_image_generator, batch_size=batch_size)

    plot_two_images_array(test_X, test_Y, 'Inference - RGB, NDVI', 0, cfg.SAVE_FIGURE_PATH)

    model = Model(model_name=cfg.MODEL_NAME, input_dim=test_X.shape[0])
    model.compile(loss='mean_absolute_error', metrics=RootMeanSquaredError())
    model.load_weights(model_path)
    model.summary()

    predict = predict_function(model, test_X, test_Y.shape, batch_size)
    lossfunc = model.evaluate(zip(test_X_ds, test_Y_ds), verbose=1)
    calculate_statistics(test_Y, predict)

    np.save(cfg.OUTPUT_DEFAULT_PATH.joinpath("predict.npy"), predict, allow_pickle=True)
    plot_three_images_array(test_X, test_Y, predict, 'Inference - RGB, NDVI, Predict', 0, cfg.SAVE_FIGURE_PATH)
    save_result_image(test_X, test_Y, predict, output_compare=True, save_image_path=cfg.SAVE_IMAGE_PATH.joinpath("inference/output"))
