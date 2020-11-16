import os

import matplotlib.pyplot as plt
import matplotlib.image
import numpy as np


def load_data():
    rgb_path = os.path.join('..', 'Jim', 'dataset',
                            '20meter', 'train_20meter_RGB.npy')
    # ndvi_path = os.path.join('..', 'Jim', 'dataset','20meter', 'train_20meter_NDVI.npy')
    rgb_image_array = np.load(rgb_path, allow_pickle=True)
    # ndvi_image_array = np.load(ndvi_path, allow_pickle=True)
    print('array shape: ', rgb_image_array.shape)
    data_array = rgb_image_array
    return data_array


def save_image(data_array, idx):
    image = data_array[idx]
    matplotlib.image.imsave('fig/image.jpg', image)
    print('image saved')


if __name__ == "__main__":
    print("load data...")
    data_array = load_data()
    save_image(data_array, 240)
