import matplotlib.image
import matplotlib.pyplot as plt
import numpy as np


class ImageDataSet():
    """
    image dataset class for NN model
    """

    def __init__(self, obj_name, data_path, save_image_path):
        self.obj_name = obj_name
        self.data_path = data_path
        self.data_raw = None
        self.data_resample = None
        self.num = self.width = self.height = self.channel = None
        self.save_image_path = save_image_path

    def load_data(self, devided_by_255=False, expand_dims=False, save_image=False):
        self.data_raw = np.load(self.data_path, mmap_mode='r', allow_pickle=True)
        # self.data_raw = np.load(self.data_path, allow_pickle=True).astype('float32')
        if devided_by_255:
            self.data_raw = self.data_raw / 255.
        if expand_dims:
            self.data_raw = np.expand_dims(self.data_raw, axis=3)
        self.num, self.height, self.width, self.channel = self.data_raw.shape
        if save_image:
            rgb_path, ndvi_path = self.output_init("raw")
            for i in range(self.num):
                if self.channel is 3:
                    matplotlib.image.imsave(rgb_path.joinpath("rgb_{0}.jpg".format(i)), self.data_raw[i], vmin=0, vmax=1)
                elif self.channel is 1:
                    img = np.squeeze(self.data_raw[i])
                    matplotlib.image.imsave(ndvi_path.joinpath("ndvi_{0}.jpg".format(i)), img, vmin=-1, vmax=1, cmap=plt.get_cmap('jet'))
        print('Data {0} shape: {1}'.format(self.obj_name, self.data_raw.shape))

    def crop(self, top_width=0, down_width=54, left_width=35, right_width=20, save_image=False):
        if self.data_raw is not None:
            self.data_raw = self.data_raw[:, top_width:(self.height - down_width), left_width:(self.width - right_width), :]
            self.num, self.height, self.width, self.channel = self.data_raw.shape
            if save_image:
                rgb_path, ndvi_path = self.output_init("crop")
                for i in range(self.num):
                    if self.channel is 3:
                        matplotlib.image.imsave(rgb_path.joinpath("rgb_{0}.jpg".format(i)), self.data_raw[i], vmin=0, vmax=1)
                    elif self.channel is 1:
                        img = np.squeeze(self.data_raw[i])
                        matplotlib.image.imsave(ndvi_path.joinpath("ndvi_{0}.jpg".format(i)), img, vmin=-1, vmax=1, cmap=plt.get_cmap('jet'))
            print('Data {0} shape after crop: {1}'.format(self.obj_name, self.data_raw.shape))
        else:
            print('No data: load data first.')

    def resample(self, table, target_dim=(352, 480), save_image=False):
        if self.data_raw is not None:
            self.data_resample = np.zeros((table.shape[0], target_dim[0], target_dim[1], self.channel), np.float32)
            if save_image:
                rgb_path, ndvi_path = self.output_init("resample")
            for i in range(table.shape[0]):
                index, top, down, left, right = table[i]
                self.data_resample[i] = self.data_raw[index, top:down, left:right, :]
                if save_image:
                    if self.channel is 3:
                        matplotlib.image.imsave(rgb_path.joinpath("rgb_{0}.jpg".format(i)), self.data_resample[i], vmin=0, vmax=1)
                    elif self.channel is 1:
                        img = np.squeeze(self.data_resample[i])
                        matplotlib.image.imsave(ndvi_path.joinpath("ndvi_{0}.jpg".format(i)), img, vmin=-1, vmax=1, cmap=plt.get_cmap('jet'))
            print('Resample data {0} shape: {1}'.format(self.obj_name, self.data_resample.shape))
        else:
            print('No data: load data first.')

    def generate_resample_table(self, target_dim=(352, 480), multiple_factor=9):
        array_len = self.num * multiple_factor
        height_delta = self.height-target_dim[0]
        width_delta = self.width-target_dim[1]
        random_max = height_delta * width_delta
        random_array = np.random.randint(low=0, high=random_max, size=array_len)
        index_array = np.repeat(np.arange(self.num, dtype=np.uint32), multiple_factor).reshape(-1, 1)
        top_array = np.rint(np.mod(random_array, height_delta)).astype(np.uint32).reshape(-1, 1)
        down_array = top_array + target_dim[0]
        left_array = np.rint(np.mod(random_array, width_delta)).astype(np.uint32).reshape(-1, 1)
        right_array = left_array + target_dim[1]
        table = np.concatenate((index_array, top_array, down_array, left_array, right_array), axis=1)
        print('Table is generated.')
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
            matplotlib.image.imsave(self.save_image_path.joinpath("raw_{}.jpg".format(index)), self.data_raw[index], vmin=0, vmax=1)
            print('Image saved: index ', index)

    def output_init(self, prefix):
        rgb_path = self.save_image_path.joinpath("{}/rgb".format(prefix))
        ndvi_path = self.save_image_path.joinpath("{}/ndvi".format(prefix))
        rgb_path.mkdir(parents=True, exist_ok=True)
        ndvi_path.mkdir(parents=True, exist_ok=True)
        return rgb_path, ndvi_path
