import copy

import numpy as np
import tensorflow as tf
from cfgs import cfg
from sklearn.utils import shuffle
from tensorflow.keras import Sequential
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def output_init():
    train_image_path = cfg.SAVE_IMAGE_PATH.joinpath('data_aug/train/rgb')
    train_mask_path = cfg.SAVE_IMAGE_PATH.joinpath('data_aug/train/ndvi')
    validation_image_path = cfg.SAVE_IMAGE_PATH.joinpath('data_aug/val/rgb')
    validation_mask_path = cfg.SAVE_IMAGE_PATH.joinpath('data_aug/val/ndvi')
    train_image_path.mkdir(parents=True, exist_ok=True)
    train_mask_path.mkdir(parents=True, exist_ok=True)
    validation_image_path.mkdir(parents=True, exist_ok=True)
    validation_mask_path.mkdir(parents=True, exist_ok=True)
    return train_image_path, train_mask_path, validation_image_path, validation_mask_path


def train_preprocessing(train_X, train_Y, batch_size, enable_data_aug, use_imagedatagenerator, datagen_args, seed, val_split):
    train_X, train_Y = shuffle(train_X, train_Y, random_state=seed)

    if enable_data_aug:
        if use_imagedatagenerator:
            train_image_datagen = ImageDataGenerator(**datagen_args)
            train_mask_datagen = ImageDataGenerator(**datagen_args)
            validation_image_datagen = ImageDataGenerator(validation_split=val_split)
            validation_mask_datagen = ImageDataGenerator(validation_split=val_split)

            train_mask_datagen.brightness_range = None
            train_mask_datagen.channel_shift_range = 0.0

            # train_image_path, train_mask_path, validation_image_path, validation_mask_path = output_init()
            train_image_generator = train_image_datagen.flow(train_X, batch_size=1, shuffle=True, seed=seed, subset='training')
            train_mask_generator = train_mask_datagen.flow(train_Y, batch_size=1, shuffle=True, seed=seed, subset='training')
            validation_image_generator = validation_image_datagen.flow(train_X, batch_size=1, shuffle=True, seed=seed, subset='validation')
            validation_mask_generator = validation_mask_datagen.flow(train_Y, batch_size=1, shuffle=True, seed=seed, subset='validation')

            train_generator = zip(train_image_generator, train_mask_generator)
            validation_generator = zip(validation_image_generator, validation_mask_generator)

            def keras_image_generator(generator):
                for data in generator:
                    data[0].shape = data[0].shape[1:]
                    data[1].shape = data[1].shape[1:]
                    yield data

            output_signature = (tf.TensorSpec(shape=train_X.shape[1:], dtype=tf.float32), tf.TensorSpec(shape=train_Y.shape[1:], dtype=tf.float32))
            train_ds = tf.data.Dataset.from_generator(lambda: keras_image_generator(train_generator), output_signature=output_signature)
            validation_ds = tf.data.Dataset.from_generator(lambda: keras_image_generator(validation_generator), output_signature=output_signature)

            train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
            validation_ds = validation_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

            return train_ds, validation_ds
        else:
            validation_split_count = int(train_X.shape[0] * (1 - val_split))
            train_ds = tf.data.Dataset.from_tensor_slices((train_X, train_Y)).take(validation_split_count)
            validation_ds = tf.data.Dataset.from_tensor_slices((train_X, train_Y)).skip(validation_split_count)

            data_augmentation_layer = [
                train_preprocessing.RandomFlip(mode="horizontal_and_vertical", seed=seed),
                train_preprocessing.RandomRotation(factor=(cfg.DATAGEN_ARGS["rotation_range"] / 360), fill_mode=cfg.DATAGEN_ARGS["fill_mode"], interpolation="bilinear", seed=seed),
                train_preprocessing.RandomZoom(height_factor=cfg.DATAGEN_ARGS["zoom_range"], fill_mode=cfg.DATAGEN_ARGS["fill_mode"], interpolation="bilinear", seed=seed),
            ]
            image_model_map = Sequential(copy.deepcopy(data_augmentation_layer))
            mask_model_map = Sequential(copy.deepcopy(data_augmentation_layer))

            rng = tf.random.Generator.from_seed(seed=seed, alg='philox')

            def tf_image_map(image):
                seed_table = rng.make_seeds(6)[0]
                image = tf.image.stateless_random_brightness(image, max_delta=0.1, seed=seed_table[0:2])
                image = tf.image.stateless_random_contrast(image, lower=0.9, upper=1.0, seed=seed_table[2:4])
                image = tf.image.stateless_random_saturation(image, lower=0.9, upper=1.0, seed=seed_table[4:6])
                return image

            train_ds = train_ds.shuffle(128, seed=seed)
            validation_ds = validation_ds.shuffle(128, seed=seed)

            train_ds = train_ds.map(lambda image, mask: (tf_image_map(image), mask), num_parallel_calls=tf.data.AUTOTUNE)

            train_ds = train_ds.batch(batch_size)
            validation_ds = validation_ds.batch(batch_size)

            train_ds = train_ds.map(lambda image, mask: (image_model_map(image, training=True), mask_model_map(mask, training=True)), num_parallel_calls=1)

            train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
            validation_ds = validation_ds.prefetch(tf.data.AUTOTUNE)

            return train_ds, validation_ds
    else:
        validation_split_count = int(train_X.shape[0] * (1 - val_split))
        train_ds = tf.data.Dataset.from_tensor_slices((train_X, train_Y)).take(validation_split_count)
        validation_ds = tf.data.Dataset.from_tensor_slices((train_X, train_Y)).skip(validation_split_count)

        train_ds = train_ds.shuffle(128, seed=seed).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        validation_ds = validation_ds.shuffle(128, seed=seed).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        return train_ds, validation_ds


def test_precessing(test_X, test_Y, batch_size):
    test_ds = tf.data.Dataset.from_tensor_slices((test_X, test_Y))
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return test_ds
