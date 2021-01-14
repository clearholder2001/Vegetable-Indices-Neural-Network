import os

import matplotlib.pyplot as plt

from configs import cfgs
from dataset import *

if __name__ == "__main__":
    rbg_obj = DataObject('RGB ', cfgs.TRAIN_RGB_PATH)
    ndvi_obj = DataObject('NDVI', cfgs.TRAIN_NDVI_PATH)
    rbg_obj.load_data(devided_by_255=True, expand_dims=False)
    ndvi_obj.load_data(devided_by_255=False, expand_dims=True)
    rbg_obj.crop()
    ndvi_obj.crop()
    table = rbg_obj.generate_resample_table(multiple_factor=cfgs.RESAMPLE_MULTIPLE_FACTOR)
    rbg_obj.resample(table)
    ndvi_obj.resample(table)
    rgb_array = rbg_obj.get_data_resample()
    ndvi_array = ndvi_obj.get_data_resample()
    plot_two_images_array(rgb_array, ndvi_array, 'dataset_test', 0)
