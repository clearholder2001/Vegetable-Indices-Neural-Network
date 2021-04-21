from pathlib import Path

import numpy as np

init_path = Path("../Jim/dataset/20meter")
include_sub_path = False
regex = "*.npy"


def list_file(path):
    if include_sub_path:
        file_list = list(path.rglob(regex))
    else:
        file_list = list(path.glob(regex))

    print("Found file:")
    for filename in file_list:
        print(filename)

    # remove filename ending with "_f32"
    for filename in file_list:
        if str(filename.stem).endswith("_f32"):
            print("Exclude file: {0}".format(filename))
            file_list.remove(filename)

    return file_list


def do_task(filename):
    array_raw = np.load(filename, mmap_mode='r', allow_pickle=True)
    print("Load file: {0}, shape={1}, dtype= {2}, max={3}, min={4}".format(filename, array_raw.shape, array_raw.dtype, np.max(array_raw), np.min(array_raw)))

    if filename.match("*/single/*"):
        array_raw = np.expand_dims(array_raw, axis=0)
        print("Change shape: {0}".format(array_raw.shape))

    if str(filename.stem).endswith("_RGB"):
        array_after = array_raw.astype('float32', copy=True) / 255.
    else:
        array_after = array_raw.astype('float32', copy=True)
        array_after = np.expand_dims(array_after, axis=3)

    save_filename = filename.parent.joinpath(filename.stem + "_f32.npy")
    np.save(save_filename, array_after, allow_pickle=True)
    print("Save file: {0}, shape={1}, dtype= {2}, max={3}, min={4}".format(save_filename, array_after.shape, array_after.dtype, np.max(array_after), np.min(array_after)))


if __name__ == "__main__":
    file_list = list_file(init_path)

    for file in file_list:
        do_task(file)
