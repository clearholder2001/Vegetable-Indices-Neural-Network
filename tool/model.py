import math

from tensorflow.keras.layers import (Activation, BatchNormalization,
                                     Concatenate, Conv2D, Conv2DTranspose,
                                     Input, MaxPooling2D)
from tensorflow.keras.models import Model


def Conv2D_block(input, element_num, filters, kernel_size, padding, kernel_initializer, activation, block_name):
    input_layer = input
    for i in range(1, element_num+1):
        layer = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_initializer, name="{0}_conv{1}".format(block_name, i))(input_layer)
        layer = BatchNormalization(name="{0}_bn{1}".format(block_name, i))(layer)
        layer = Activation(activation, name="{0}_ac{1}".format(block_name, i))(layer)
        input_layer = layer
    return layer


def unet_default(model_name, input_dim):
    Input_img = Input(input_dim)

    # Setup
    filter_factor = 64
    conv2D_kernel_size = (3, 3)
    resample_size = (3, 3)
    resample_strides = (2, 2)
    padding = 'same'
    kernel_initializer = 'he_uniform'
    activation = 'relu'

    # Encoder Architecture
    # Block 1
    en_b1 = Conv2D_block(input=Input_img, element_num=2, filters=(filter_factor*math.pow(2, 0)), kernel_size=conv2D_kernel_size, padding=padding, kernel_initializer=kernel_initializer, activation=activation, block_name='encoder_b1')
    en_b1_pooling = MaxPooling2D(resample_size, strides=resample_strides, padding=padding, name='block1_pool')(en_b1)

    # Block 2
    en_b2 = Conv2D_block(input=en_b1_pooling, element_num=2, filters=(filter_factor*math.pow(2, 1)), kernel_size=conv2D_kernel_size, padding=padding, kernel_initializer=kernel_initializer, activation=activation, block_name='encoder_b2')
    en_b2_pooling = MaxPooling2D(resample_size, strides=resample_strides, padding=padding, name='block2_pool')(en_b2)

    # Block 3
    en_b3 = Conv2D_block(input=en_b2_pooling, element_num=3, filters=(filter_factor*math.pow(2, 2)), kernel_size=conv2D_kernel_size, padding=padding, kernel_initializer=kernel_initializer, activation=activation, block_name='encoder_b3')
    en_b3_pooling = MaxPooling2D(resample_size, strides=resample_strides, padding=padding, name='block3_pool')(en_b3)

    # Block 4
    # en_b4 = Conv2D_block(input=en_b3_pooling, element_num=2, filters=(filter_factor*math.pow(2, 3)), kernel_size=conv2D_kernel_size, padding=padding, kernel_initializer=kernel_initializer, activation=activation, block_name='encoder_b4')
    # en_b4_pooling = MaxPooling2D(resample_size, strides=resample_strides, padding=padding, name='block4_pool')(en_b4)

    encoder = en_b3_pooling

    # Decoder Architecture
    # Block 1
    # de_b1 = Conv2DTranspose(filters=(filter_factor*math.pow(2, 2)), kernel_size=resample_size, strides=resample_strides, padding=padding, name='decoder_b1_convT')(encoder)
    # de_b1 = Concatenate(axis=3)([en_b3, de_b1])
    # de_b1 = Conv2D_block(input=de_b1, element_num=3, filters=(filter_factor*math.pow(2, 2)), kernel_size=conv2D_kernel_size, padding=padding, kernel_initializer=kernel_initializer, activation=activation, block_name='decoder_b1')

    # Block 2
    de_b2 = Conv2DTranspose(filters=(filter_factor*math.pow(2, 2)), kernel_size=resample_size, strides=resample_strides, padding=padding, name='decoder_b2_convT')(encoder)
    de_b2 = Concatenate(axis=3)([en_b3, de_b2])
    de_b2 = Conv2D_block(input=de_b2, element_num=3, filters=(filter_factor*math.pow(2, 2)), kernel_size=conv2D_kernel_size, padding=padding, kernel_initializer=kernel_initializer, activation=activation, block_name='decoder_b2')

    # Block 3
    de_b3 = Conv2DTranspose(filters=(filter_factor*math.pow(2, 1)), kernel_size=resample_size, strides=resample_strides, padding=padding, name='decoder_b3_convT')(de_b2)
    de_b3 = Concatenate(axis=3)([en_b2, de_b3])
    de_b3 = Conv2D_block(input=de_b3, element_num=2, filters=(filter_factor*math.pow(2, 1)), kernel_size=conv2D_kernel_size, padding=padding, kernel_initializer=kernel_initializer, activation=activation, block_name='decoder_b3')

    # Block 4
    de_b4 = Conv2DTranspose(filters=(filter_factor*math.pow(2, 0)), kernel_size=resample_size, strides=resample_strides, padding=padding, name='decoder_b4_convT')(de_b3)
    de_b4 = Concatenate(axis=3)([en_b1, de_b4])
    de_b4 = Conv2D_block(input=de_b4, element_num=2, filters=(filter_factor*math.pow(2, 0)), kernel_size=conv2D_kernel_size, padding=padding, kernel_initializer=kernel_initializer, activation=activation, block_name='decoder_b4')

    decoder = de_b4

    # Output
    output = Conv2D(filters=1, kernel_size=resample_size, padding=padding, kernel_initializer='glorot_uniform', name='output_conv')(decoder)

    model = Model(inputs=Input_img, outputs=output, name=model_name)
    return model
