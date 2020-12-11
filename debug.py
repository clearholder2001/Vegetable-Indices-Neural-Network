import os

import matplotlib.pyplot as plt

from dataset import DataObject
from configs import cfgs


def plot_multiimages(images1, images2, title, idx, num=36):
    plt.gcf().set_size_inches(9, 9)
    if num > 36:
        num = 36
    for i in range(0, int(num/2)):
        ax = plt.subplot(6, 6, 1+i)
        ax.imshow(images1[idx+i], vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
    for i in range(0, int(num/2)):
        ax = plt.subplot(6, 6, int(num/2)+1+i)
        ax.imshow(images2[idx+i], vmin=-1, vmax=1, cmap=plt.get_cmap('jet'))
        ax.set_xticks([])
        ax.set_yticks([])
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    plt.savefig('./fig/' + title + '.png')


if __name__ == "__main__":
    rbg_obj = DataObject('RGB ', cfgs.RGB_PATH)
    ndvi_obj = DataObject('NDVI', cfgs.NDVI_PATH)
    rbg_obj.get_data_raw()
    ndvi_obj.get_data_raw()
    rbg_obj.load_data(devided_by_255=True, expand_dims=False)
    ndvi_obj.load_data(devided_by_255=False, expand_dims=True)
    rbg_obj.crop()
    ndvi_obj.crop()
    table = rbg_obj.generate_resample_table(multiple_factor=cfgs.RESAMPLE_MULTIPLE_FACTOR)
    rbg_obj.resample(table)
    ndvi_obj.resample(table)
    rgb_array = rbg_obj.get_data_resample()
    ndvi_array = ndvi_obj.get_data_resample()
    plot_multiimages(rgb_array, ndvi_array, 'dataset_test', 0*9)
