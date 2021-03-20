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


def save_result_image(test_X, test_Y, predict, output_compare=True):
    assert test_X.shape[0] == test_Y.shape[0] == predict.shape[0], 'Length inconsistent: test_X, test_Y, preditc'
    path = './fig/inference/'
    print("Saving result...", end='')

    if output_compare:
        fig, axs = plt.subplots(1, 3)
        fig.set_size_inches(12, 4)
        plt.setp(axs, xticks=[], yticks=[])
        axs[0].set_title("RGB")
        img1 = axs[0].imshow(test_X[0], vmin=0, vmax=1)
        axs[1].set_title("NDVI")
        img2 = axs[1].imshow(test_Y[0], vmin=-1, vmax=1, cmap=plt.get_cmap('jet'))
        axs[2].set_title("Predict")
        img3 = axs[2].imshow(predict[0], vmin=-1, vmax=1, cmap=plt.get_cmap('jet'))

    for i in range(test_X.shape[0]):
        matplotlib.image.imsave(path + 'rgb/rgb_{0}.jpg'.format(i), test_X[i])
        matplotlib.image.imsave(path + 'ndvi/ndvi_{0}.jpg'.format(i), np.squeeze(test_Y[i]), cmap=plt.get_cmap('jet'))
        matplotlib.image.imsave(path + 'predict/predict_{0}.jpg'.format(i), np.squeeze(predict[i]), cmap=plt.get_cmap('jet'))
        if output_compare:
            img1.set_data(test_X[i])
            img2.set_data(test_Y[i])
            img3.set_data(predict[i])
            fig.suptitle("Compare #" + str(i), fontsize=24)
            fig.tight_layout()
            fig.savefig(path + 'compare/compare_{0}.jpg'.format(i))
    plt.close()
    print("Done")


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
