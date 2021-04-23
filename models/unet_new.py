import math

from tensorflow.keras.layers import (Activation, Concatenate, Conv2D,
                                     Conv2DTranspose, Input, MaxPooling2D)
from tensorflow.keras.models import Model


def Conv2D_block(input, element_num, filters, kernel_size, padding, kernel_initializer, activation, block_name):
    input_layer = input
    for i in range(1, element_num+1):
        layer = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_initializer, name="{0}_conv{1}".format(block_name, i))(input_layer)
        layer = Activation(activation, name="{0}_ac{1}".format(block_name, i))(layer)
        input_layer = layer
    return layer


def unet_new(model_name):
    Input_img = Input(shape=(None, None, 3))

    # Setup
    filter_factor = 8
    kernel_size = (3, 3)
    padding = 'same'
    kernel_initializer = 'he_uniform'
    activation = 'relu'

    # Encoder Architecture
    # Block 1
    en_b1 = Conv2D_block(input=Input_img, element_num=3, filters=(filter_factor*math.pow(2, 0)), kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_initializer, activation=activation, block_name='encoder_b1')
    en_b1_pooling = MaxPooling2D((2, 2), strides=(2, 2), padding=padding, name='block1_pool')(en_b1)

    # Block 2
    en_b2 = Conv2D_block(input=en_b1_pooling, element_num=3, filters=(filter_factor*math.pow(2, 1)), kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_initializer, activation=activation, block_name='encoder_b2')
    en_b2_pooling = MaxPooling2D((2, 2), strides=(2, 2), padding=padding, name='block2_pool')(en_b2)

    # Block 3
    en_b3 = Conv2D_block(input=en_b2_pooling, element_num=3, filters=(filter_factor*math.pow(2, 2)), kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_initializer, activation=activation, block_name='encoder_b3')
    en_b3_pooling = MaxPooling2D((2, 2), strides=(2, 2), padding=padding, name='block3_pool')(en_b3)

    # Block 4
    en_b4 = Conv2D_block(input=en_b3_pooling, element_num=3, filters=(filter_factor*math.pow(2, 3)), kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_initializer, activation=activation, block_name='encoder_b4')
    en_b4_pooling = MaxPooling2D((2, 2), strides=(2, 2), padding=padding, name='block4_pool')(en_b4)

    # Block 5
    en_b5 = Conv2D_block(input=en_b4_pooling, element_num=3, filters=(filter_factor*math.pow(2, 4)), kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_initializer, activation=activation, block_name='encoder_b5')

    encoder = en_b5

    # Decoder Architecture
    # Block 1
    de_b1 = Conv2DTranspose(filters=(filter_factor*math.pow(2, 3)), kernel_size=kernel_size, strides=(2, 2), padding=padding, kernel_initializer=kernel_initializer, name='decoder_b1_convT1')(encoder)
    de_b1 = Activation(activation, name='decoder_b1_convT1_ac')(de_b1)
    de_b1 = Concatenate(axis=3)([en_b4, de_b1])
    de_b1 = Conv2D_block(input=de_b1, element_num=2, filters=(filter_factor*math.pow(2, 3)), kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_initializer, activation=activation, block_name='decoder_b1')

    # Block 2
    de_b2 = Conv2DTranspose(filters=(filter_factor*math.pow(2, 2)), kernel_size=kernel_size, strides=(2, 2), padding=padding, kernel_initializer=kernel_initializer, name='decoder_b2_convT1')(de_b1)
    de_b2 = Activation(activation, name='decoder_b2_convT1_ac')(de_b2)
    de_b2 = Concatenate(axis=3)([en_b3, de_b2])
    de_b2 = Conv2D_block(input=de_b2, element_num=2, filters=(filter_factor*math.pow(2, 2)), kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_initializer, activation=activation, block_name='decoder_b2')

    # Block 3
    de_b3 = Conv2DTranspose(filters=(filter_factor*math.pow(2, 1)), kernel_size=kernel_size, strides=(2, 2), padding=padding, kernel_initializer=kernel_initializer, name='decoder_b3_convT1')(de_b2)
    de_b3 = Activation(activation, name='decoder_b3_convT1_ac')(de_b3)
    de_b3 = Concatenate(axis=3)([en_b2, de_b3])
    de_b3 = Conv2D_block(input=de_b3, element_num=2, filters=(filter_factor*math.pow(2, 1)), kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_initializer, activation=activation, block_name='decoder_b3')

    # Block 4
    de_b4 = Conv2DTranspose(filters=(filter_factor*math.pow(2, 0)), kernel_size=kernel_size, strides=(2, 2), padding=padding, kernel_initializer=kernel_initializer, name='decoder_b4_convT1')(de_b3)
    de_b4 = Activation(activation, name='decoder_b4_convT1_ac')(de_b4)
    de_b4 = Concatenate(axis=3)([en_b1, de_b4])
    de_b4 = Conv2D_block(input=de_b4, element_num=2, filters=(filter_factor*math.pow(2, 0)), kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_initializer, activation=activation, block_name='decoder_b4')

    decoder = Conv2D(filters=1, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_initializer, name='output_conv')(de_b4)

    model = Model(inputs=Input_img, outputs=decoder, name=model_name)
    return model
