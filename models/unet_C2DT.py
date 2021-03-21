from cfgs import cfg
from tensorflow.keras.layers import (Activation, BatchNormalization,
                                     Concatenate, Conv2D, Conv2DTranspose,
                                     Input, MaxPooling2D)
from tensorflow.keras.models import Model


def unet_C2DT(model_name):
    Input_img = Input(shape=cfg.INPUT_LAYER_DIM)

    # Setup
    activation = 'relu'

    # Encoding Architecture
    # Block 1
    x1 = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal', name='block1_conv1')(Input_img)
    x1 = Activation(activation, name='block1_ac1')(x1)
    x1 = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal', name='block1_conv2')(x1)
    x1 = Activation(activation, name='block1_ac2')(x1)
    x1 = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal', name='block1_conv3')(x1)
    x1 = Activation(activation, name='block1_ac3')(x1)
    x1d = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block1_pool')(x1)

    # Block 2
    x2 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal', name='block2_conv1')(x1d)
    x2 = Activation(activation, name='block2_ac1')(x2)
    x2 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal', name='block2_conv2')(x2)
    x2 = Activation(activation, name='block2_ac2')(x2)
    x2 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal', name='block2_conv3')(x2)
    x2 = Activation(activation, name='block2_ac3')(x2)
    x2d = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block2_pool')(x2)

    # Block 3
    x3 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', name='block3_conv1')(x2d)
    x3 = Activation(activation, name='block3_ac1')(x3)
    x3 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', name='block3_conv2')(x3)
    x3 = Activation(activation, name='block3_ac2')(x3)
    x3 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', name='block3_conv3')(x3)
    x3 = Activation(activation, name='block3_ac3')(x3)
    x3d = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block3_pool')(x3)

    # Block 4
    x4 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', name='block4_conv1')(x3d)
    x4 = Activation(activation, name='block4_ac1')(x4)
    x4 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', name='block4_conv2')(x4)
    x4 = Activation(activation, name='block4_ac2')(x4)
    x4 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', name='block4_conv3')(x4)
    x4 = Activation(activation, name='block4_ac3')(x4)
    x4d = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block4_pool')(x4)

    # Block 5
    x5 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', name='block5_conv1')(x4d)
    x5 = Activation(activation, name='block5_ac1')(x5)
    x5 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', name='block5_conv2')(x5)
    x5 = Activation(activation, name='block5_ac2')(x5)
    x5 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', name='block5_conv3')(x5)
    x5 = Activation(activation, name='block5_ac3')(x5)

    encoded = x5

    # Decoding Architecture
    # Block 6
    x6 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal', name='block6_convT1')(encoded)
    x6 = Activation(activation, name='block6_ac1')(x6)
    x6 = Concatenate(axis=3)([x4, x6])
    x6 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', name='block6_conv2')(x6)
    x6 = Activation(activation, name='block6_ac2')(x6)
    x6 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', name='block6_conv3')(x6)
    x6 = Activation(activation, name='block6_ac3')(x6)

    # Block 7
    x7 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal', name='block7_convT1')(x6)
    x7 = Activation(activation, name='block7_ac1')(x7)
    x7 = Concatenate(axis=3)([x3, x7])
    x7 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', name='block7_conv2')(x7)
    x7 = Activation(activation, name='block7_ac2')(x7)
    x7 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', name='block7_conv3')(x7)
    x7 = Activation(activation, name='block7_ac3')(x7)

    # Block 8
    x8 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal', name='block8_convT1')(x7)
    x8 = Activation(activation, name='block8_ac1')(x8)
    x8 = Concatenate(axis=3)([x2, x8])
    x8 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal', name='block8_conv2')(x8)
    x8 = Activation(activation, name='block8_ac2')(x8)
    x8 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal', name='block8_conv3')(x8)
    x8 = Activation(activation, name='block8_ac3')(x8)

    # Block 9
    x9 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal', name='block9_convT1')(x8)
    x9 = Activation(activation, name='block9_ac1')(x9)
    x9 = Concatenate(axis=3)([x1, x9])
    x9 = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal', name='block9_conv2')(x9)
    x9 = Activation(activation, name='block9_ac2')(x9)
    x9 = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal', name='block9_conv3')(x9)
    x9 = Activation(activation, name='block9_ac3')(x9)
    #x9 = Activation(activation='tanh', name='block9_ac3')(x9)

    decoded = Conv2D(1, (3, 3), activation='tanh', padding='same', kernel_initializer='glorot_normal', name='block9_output')(x9)
    #decoded = Lambda(lambda x: mean(x, axis=3)[:, :, :, None])(x9)

    model = Model(inputs=Input_img, outputs=decoded, name=model_name)
    return model
