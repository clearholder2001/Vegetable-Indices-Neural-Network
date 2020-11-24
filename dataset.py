import os

import matplotlib.image
import matplotlib.pyplot as plt
import numpy as np


class DataObject():
    """
    Input data for NN model
    """

    def __init__(self, path_name):
        self.path_name = path_name
        self.data = None
        self.amount = self.width = self.height = self.channel = None

    def load_data(self, devided_by_255=True, expand_dims=False):
        self.data = np.load(self.path_name, allow_pickle=True)
        if devided_by_255:
            self.data = self.data.astype('float32') / 255.
        if expand_dims:
            self.data = np.expand_dims(self.data, axis=3)
        self.amount, self.height, self.width, self.channel = self.data.shape
        print('Data shape: ', self.data.shape)

    def crop(self, top_width=0, down_width=52, left_width=23, right_width=23):
        if self.data is not None:
            self.data = self.data[:, top_width:(
                self.height - down_width), left_width:(self.width - right_width), :]
            print('Data shape after crop: ', self.data.shape)
        else:
            print('No data: load data first.')

    def get_data(self):
        return self.data

    def save_single_image(self, index=0):
        if self.data is not None:
            matplotlib.image.imsave('fig/image.jpg', self.data[index])
            print('Image saved: index ', index)
