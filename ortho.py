import copy

import numpy as np
import rasterio
from skimage.util import view_as_blocks


filename = "odm_orthophoto_small.tif"

raw_image_array = None
clip_image_array = None
patch_image_array = None

patch_height = patch_width = 224

if __name__ == "__main__":
    with rasterio.open(filename, 'r') as ds:
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
    clip_image_array = np.transpose(clip_image_array, [1, 2, 0])
    clip_image_array = clip_image_array[clip_extent_height_min:clip_extent_height_max, clip_extent_width_min:clip_extent_width_max, 0:3]
    print(clip_image_array.shape)

    block_image_array = view_as_blocks(clip_image_array, (patch_height, patch_width, 3))
    print(block_image_array.shape)

    patch_image_array = block_image_array.reshape((-1, patch_height, patch_width, 3))
