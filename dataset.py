import os
from time import time

import matplotlib.image
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class DataObject():
    """
    data object for NN model
    """

    def __init__(self, obj_name, path_name):
        self.path_name = path_name
        self.obj_name = obj_name
        self.data_raw = None
        self.data_resample = None
        self.num = self.width = self.height = self.channel = None

    def load_data(self, devided_by_255=True, expand_dims=False, save_image=False):
        self.data_raw = np.load(self.path_name, allow_pickle=True)
        if devided_by_255:
            self.data_raw = self.data_raw.astype('float32') / 255.
        if expand_dims:
            self.data_raw = np.expand_dims(self.data_raw, axis=3)
        self.num, self.height, self.width, self.channel = self.data_raw.shape
        if save_image:
            for i in range(self.num):
                if self.channel is 3:
                    matplotlib.image.imsave('fig/raw/rgb/rgb_{0}.jpg'.format(i), self.data_raw[i])
                elif self.channel is 1:
                    img = np.squeeze(self.data_raw[i])
                    matplotlib.image.imsave('fig/raw/ndvi/ndvi_{0}.jpg'.format(i), img, cmap=plt.get_cmap('gray'))
        print('Data {0} shape: {1}'.format(self.obj_name, self.data_raw.shape))

    def crop(self, top_width=0, down_width=52, left_width=23, right_width=23, save_image=False):
        if self.data_raw is not None:
            self.data_raw = self.data_raw[:, top_width:(self.height - down_width), left_width:(self.width - right_width), :]
            self.num, self.height, self.width, self.channel = self.data_raw.shape
            if save_image:
                for i in range(self.num):
                    if self.channel is 3:
                        matplotlib.image.imsave('fig/crop/rgb/rgb_{0}.jpg'.format(i), self.data_raw[i])
                    elif self.channel is 1:
                        img = np.squeeze(self.data_raw[i])
                        matplotlib.image.imsave('fig/crop/ndvi/ndvi_{0}.jpg'.format(i), img, cmap=plt.get_cmap('gray'))
            print('Data {0} shape after crop: {1}'.format(self.obj_name, self.data_raw.shape))
        else:
            print('No data: load data first.')

    def resample(self, table, target_size=(352, 480), save_image=False):
        if self.data_raw is not None:
            self.data_resample = np.zeros((table.shape[0], target_size[0], target_size[1], self.channel), np.float32)
            for i in range(table.shape[0]):
                index, top, down, left, right = table[i]
                self.data_resample[i] = self.data_raw[index, top:down, left:right, :]
                if save_image:
                    if self.channel is 3:
                        matplotlib.image.imsave('fig/resample/rgb/0/{0}.jpg'.format(i), self.data_resample[i])
                    elif self.channel is 1:
                        img = np.squeeze(self.data_resample[i])
                        matplotlib.image.imsave('fig/resample/ndvi/0/{0}.jpg'.format(i), img, cmap=plt.get_cmap('gray'))
            print('Resample data {0} shape: {1}'.format(self.obj_name, self.data_resample.shape))
        else:
            print('No data: load data first.')

    def generate_resample_table(self, target_size=(352, 480), multiple_factor=9):
        np.random.seed(int(time()))
        array_len = self.num * multiple_factor
        index_array = np.repeat(np.arange(self.num, dtype=np.uint32), multiple_factor).reshape(-1, 1)
        top_array = np.rint(np.random.rand(array_len) * (self.height-target_size[0])).astype(np.uint32).reshape(-1, 1)
        down_array = top_array + target_size[0]
        left_array = np.rint(np.random.rand(array_len) * (self.width-target_size[1])).astype(np.uint32).reshape(-1, 1)
        right_array = left_array + target_size[1]
        table = np.concatenate((index_array, top_array, down_array, left_array, right_array), axis=1)
        print('Table is generated')
        return table

    def get_data_raw(self):
        if self.data_raw is not None:
            return self.data_raw
        else:
            print('No data: load data first.')

    def get_data_resample(self):
        if self.data_resample is not None:
            return self.data_resample
        else:
            print('No data: resample data first.')

    def save_single_image(self, index=0):
        if self.data_raw is not None:
            matplotlib.image.imsave('fig/image.jpg', self.data_raw[index])
            print('Image saved: index ', index)


def plot_two_images_array(images1, images2, title, idx):
    fig, axs = plt.subplots(4, 4)
    fig.set_size_inches(8, 6)
    plt.setp(axs, xticks=[], yticks=[])
    for i in range(4):
        axs[i, 0].imshow(images1[idx+i], vmin=0, vmax=1)
        axs[i, 1].imshow(images2[idx+i], vmin=-1, vmax=1, cmap=plt.get_cmap('jet'))
        axs[i, 2].imshow(images1[idx+4+i], vmin=0, vmax=1)
        axs[i, 3].imshow(images2[idx+4+i], vmin=-1, vmax=1, cmap=plt.get_cmap('jet'))
    fig.suptitle(title, fontsize=24)
    fig.tight_layout()
    fig.savefig('./fig/' + title + '.png')
    plt.close(fig)


def plot_three_images_array(images1, images2, images3, title, idx):
    fig, axs = plt.subplots(4, 3)
    fig.set_size_inches(12, 13)
    plt.setp(axs, xticks=[], yticks=[])
    for i in range(4):
        axs[i, 0].imshow(images1[idx+i], vmin=0, vmax=1)
        axs[i, 1].imshow(images2[idx+i], vmin=-1, vmax=1, cmap=plt.get_cmap('jet'))
        axs[i, 2].imshow(images3[idx+i], vmin=-1, vmax=1, cmap=plt.get_cmap('jet'))
    fig.suptitle(title, fontsize=24)
    fig.tight_layout()
    fig.savefig('./fig/' + title + '.png')
    plt.close(fig)
