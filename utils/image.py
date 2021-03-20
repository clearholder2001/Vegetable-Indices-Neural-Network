import os

import matplotlib.image
import matplotlib.pyplot as plt
import numpy as np


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


def load_data():
    rgb_path = os.path.join('..', 'Jim', 'dataset', '20meter', 'train_20meter_RGB.npy')
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


def image_viewer(data_array, idx):
    print("load data...")
    data_array = load_data()
    save_image(data_array, 240)
