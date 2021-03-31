from pathlib import Path

import matplotlib.image
import matplotlib.pyplot as plt
import numpy as np


def plot_two_images_array(images1, images2, title, idx, save_figure_path):
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
    fig.savefig(save_figure_path.joinpath("{}.jpg".format(title)))
    plt.close(fig)


def plot_three_images_array(images1, images2, images3, title, idx, save_figure_path):
    fig, axs = plt.subplots(4, 3)
    fig.set_size_inches(12, 13)
    plt.setp(axs, xticks=[], yticks=[])
    for i in range(4):
        axs[i, 0].imshow(images1[idx+i], vmin=0, vmax=1)
        axs[i, 1].imshow(images2[idx+i], vmin=-1, vmax=1, cmap=plt.get_cmap('jet'))
        axs[i, 2].imshow(images3[idx+i], vmin=-1, vmax=1, cmap=plt.get_cmap('jet'))
    fig.suptitle(title, fontsize=24)
    fig.tight_layout()
    fig.savefig(save_figure_path.joinpath("{}.jpg".format(title)))
    plt.close(fig)


def save_result_image(test_X, test_Y, predict, output_compare=True, save_image_path=None):
    assert test_X.shape[0] == test_Y.shape[0] == predict.shape[0], 'Length inconsistent: test_X, test_Y, preditc'

    rgb_path = save_image_path.joinpath("rgb")
    ndvi_path = save_image_path.joinpath("ndvi")
    predict_path = save_image_path.joinpath("predict")
    rgb_path.mkdir(parents=True, exist_ok=True)
    ndvi_path.mkdir(parents=True, exist_ok=True)
    predict_path.mkdir(parents=True, exist_ok=True)
    if output_compare:
        compare_path = save_image_path.joinpath("comparison")
        compare_path.mkdir(parents=True, exist_ok=True)

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
        matplotlib.image.imsave(rgb_path.joinpath("rgb_{}.jpg".format(i)), test_X[i])
        matplotlib.image.imsave(ndvi_path.joinpath("ndvi_{}.jpg".format(i)), np.squeeze(test_Y[i]), vmin=-1, vmax=1, cmap=plt.get_cmap('jet'))
        matplotlib.image.imsave(predict_path.joinpath("predict_{}.jpg".format(i)), np.squeeze(predict[i]), vmin=-1, vmax=1, cmap=plt.get_cmap('jet'))
        if output_compare:
            img1.set_data(test_X[i])
            img2.set_data(test_Y[i])
            img3.set_data(predict[i])
            fig.suptitle("Comparison #" + str(i), fontsize=24)
            fig.tight_layout()
            fig.savefig(compare_path.joinpath("comparison_{}.jpg".format(i)))
    plt.close()
    print("Done")
