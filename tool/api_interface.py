import copy
import os
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import tensorflow as tf
from skimage.util import view_as_blocks
# from tensorflow.keras import mixed_precision
from tensorflow.keras.models import load_model

# from model import unet_default as Model


# os.system('nvcc -V')
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
# mixed_precision.set_global_policy('mixed_float16')

raw_image_array = None
clip_image_array = None
patch_image_array = None
predict_color = None
block_dim = None
patch_height = patch_width = 224

orthophoto_path = Path("odm_orthophoto_small.tif")
predict_path = Path("outputs/predict.jpg")
model_path = Path("outputs/trained_model_for_odm.h5")
model_save_path = Path("outputs/trained_model_xxxxx.h5")


if __name__ == "__main__":
    with rasterio.open(orthophoto_path, 'r') as ds:
        raw_image_array = ds.read()  # read all raster values
        print(ds.profile)

    print(raw_image_array.shape)  # this is a 3D numpy array, with dimensions [band, height, width]
    _, raw_height, raw_width = raw_image_array.shape

    clip_height = int(np.floor(raw_height / patch_height) * patch_height)
    clip_width = int(np.floor(raw_width / patch_width) * patch_width)

    clip_extent_height_min = int((raw_height - clip_height) / 2)
    clip_extent_height_max = clip_extent_height_min + clip_height
    clip_extent_width_min = int((raw_width - clip_width) / 2)
    clip_extent_width_max = clip_extent_width_min + clip_width

    clip_image_array = copy.deepcopy(raw_image_array)
    clip_image_array = clip_image_array.transpose([1, 2, 0])
    clip_image_array = clip_image_array[clip_extent_height_min:clip_extent_height_max, clip_extent_width_min:clip_extent_width_max, 0:3]
    print(clip_image_array.shape)

    block_image_array = view_as_blocks(clip_image_array, (patch_height, patch_width, 3))
    block_dim = block_image_array.shape[0:3]
    print(block_image_array.shape)

    patch_image_array = block_image_array.reshape((-1, patch_height, patch_width, 3))

    # new_array = patch_image_array.reshape(block_dim+((patch_height, patch_width, 3)))

    # plt.figure()
    # plt.imshow(new_array[2,2,0,:,:,:])
    # plt.show()  # display it

    # inference
    patch_image_array = np.flip(patch_image_array, axis=3)
    test_X = patch_image_array.astype('float32') / 255.
    print(f"RGB  array shape: {test_X.shape}")

    # model = Model(model_name="unet_default", input_dim=test_X.shape[1:])
    # model.compile(loss="mean_squared_error")
    # model.load_weights(model_path)
    # model.save(model_save_path)
    # model.summary()

    model = load_model(model_path)
    # model.summary()

    batch_size = 1
    predict = model.predict(test_X, batch_size=batch_size, verbose=1)

    test_Y = (predict * 2) - 1

    test_Y_array = test_Y.reshape(block_dim+((patch_height, patch_width, 1)))
    test_Y_array = test_Y_array.transpose(0,3,1,4,2,5).reshape(clip_height, clip_width, 1)

    matplotlib.image.imsave(predict_path, test_Y_array.squeeze(), vmin=-1, vmax=1, cmap=plt.get_cmap('jet'))

    with rasterio.open(predict_path, 'r') as ds:
        predict_color = ds.read()

    raw_image_array[0:3, clip_extent_height_min:clip_extent_height_max, clip_extent_width_min:clip_extent_width_max] = predict_color
    raw_image_array[3, 0:clip_extent_height_min, :] = 0
    raw_image_array[3, clip_extent_height_max:, :] = 0
    raw_image_array[3, :, 0:clip_extent_width_min] = 0
    raw_image_array[3, :, clip_extent_width_max:] = 0

    with rasterio.open(orthophoto_path, 'r+') as ds:
        ds.write(raw_image_array)
