import matplotlib.image
import matplotlib.pyplot as plt
import numpy as np
from skimage import transform


class ImageDataSet():
    """
    image dataset class for NN model
    """

    def __init__(self, name, input_path, save_image_path):
        self.name = name
        self.input_path = input_path
        self.__image_array = None
        self.num = self.width = self.height = self.channel = None
        self.save_image_path = save_image_path

    def load_data(self, devided_by_255=False, expand_dims=False, save_image=False):
        self.__image_array = np.load(self.input_path, mmap_mode='r', allow_pickle=True)
        # For non-float32 npy file
        # self.data_raw = np.load(self.input_path, allow_pickle=True).astype('float32')

        if devided_by_255:
            self.__image_array = self.__image_array / 255.

        if expand_dims:
            self.__image_array = np.expand_dims(self.__image_array, axis=3)

        self.num, self.height, self.width, self.channel = self.__image_array.shape

        if save_image:
            self.__save_image("raw")

        print(f"Data {self.name} shape: {self.__image_array.shape}")
        return self

    def crop(self, delta=(0, 54, 35, 20), save_image=False):
        if self.__image_array is not None:
            top_delta, down_delta, left_delta, right_delta = delta
            self.__image_array = self.__image_array[:, top_delta:(self.height - down_delta), left_delta:(self.width - right_delta), :]
            self.num, self.height, self.width, self.channel = self.__image_array.shape

            if save_image:
                self.__save_image("crop")

            print(f"Data {self.name} shape after crop: {self.__image_array.shape}")
            return self
        else:
            print("No data: load data first.")

    def downscale(self, factor, save_image=False):
        if self.__image_array is not None:
            factor = (1, factor, factor, 1)
            self.__image_array = transform.downscale_local_mean(self.__image_array, factor)
            self.num, self.height, self.width, self.channel = self.__image_array.shape

            if save_image:
                self.__save_image("downscale")

            print(f"Data {self.name} shape after downscale: {self.__image_array.shape}")
            return self
        else:
            print("No data: load data first.")

    def resample(self, table, target_dim, save_image=False):
        if self.__image_array is not None:
            num = table.shape[0]
            self.resample_image_array = np.zeros((num, *target_dim, self.channel), np.float32)

            for i in range(num):
                index, top, down, left, right = table[i]
                self.resample_image_array[i] = self.__image_array[index, top:down, left:right, :]

            self.__image_array = self.resample_image_array
            self.num, self.height, self.width, self.channel = self.__image_array.shape

            if save_image:
                self.__save_image("resample")

            print(f"Data {self.name} shape after resample: {self.__image_array.shape}")
            return self
        else:
            print("No data: load data first.")

    def get_image_array(self):
        if self.__image_array is not None:
            return self.__image_array
        else:
            print("No data: load data first.")
            return None

    def __save_image(self, prefix):
        if self.channel == 3:
            type = "rgb"
            vmin, vmax = 0, 1
            cmap = None
            image_array = self.__image_array
        elif self.channel == 1:
            type = "ndvi"
            vmin, vmax = -1, 1
            cmap = plt.get_cmap('jet')
            image_array = self.__image_array[:, :, :, 0]
        else:
            return

        path = self.save_image_path.joinpath(f"{prefix}/{type}")
        path.mkdir(parents=True, exist_ok=True)

        for i in range(self.num):
            matplotlib.image.imsave(path.joinpath(f"{type}_{i}.jpg"), image_array[i], vmin=vmin, vmax=vmax, cmap=cmap)

    @staticmethod
    def generate_resample_table(image_num, multiple_factor, image_dim, target_dim):
        array_len = image_num * multiple_factor
        height_delta = image_dim[0] - target_dim[0]
        width_delta = image_dim[1] - target_dim[1]
        random_max = height_delta * width_delta

        random_array = np.random.randint(low=0, high=random_max, size=array_len)
        index_array = np.repeat(np.arange(image_num, dtype=np.uint32), multiple_factor).reshape(-1, 1)

        top_array = np.rint(np.mod(random_array, height_delta)).astype(np.uint32).reshape(-1, 1)
        down_array = top_array + target_dim[0]

        left_array = np.rint(np.mod(random_array, width_delta)).astype(np.uint32).reshape(-1, 1)
        right_array = left_array + target_dim[1]

        table = np.concatenate((index_array, top_array, down_array, left_array, right_array), axis=1)
        print("Table is generated.")
        return table
