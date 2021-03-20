from time import time

from tensorflow.keras import regularizers
from tensorflow.keras.layers import (Activation, BatchNormalization,
                                     Concatenate, Conv2D, Conv2DTranspose,
                                     Dense, Flatten, Input, Lambda,
                                     MaxPooling2D, UpSampling2D)
from tensorflow.keras.layers.experimental.preprocessing import RandomContrast
from tensorflow.keras.models import Model
from tensorflow.keras.backend import mean

from configs import cfgs


def NN_model_1():
    X_input = Input((51, 51, 3))
    # stage 1
    X = Conv2D(4, (3, 3), strides=(1, 1), name="conv1", padding="same")(X_input)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(4, (3, 3), strides=(1, 1), name="conv1_2", padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X)
    # stage 2
    X = Conv2D(8, (3, 3), strides=(1, 1), name="conv2", padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(8, (3, 3), strides=(1, 1), name="conv2_2", padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X)

    flatten = Flatten()(X)
    fcn = Dense(4096)(flatten)
    fcn = BatchNormalization()(fcn)
    fcn = Activation('relu')(fcn)
    fcn = Dense(1)(fcn)
    fcn = Activation('tanh')(fcn)
    model = Model(X_input, fcn)
    return model


def AE_model_1():
    # Defining our Image denoising autoencoder
    Input_img = Input(shape=(960, 1280, 3))

    # encoding architecture
    x1 = Conv2D(16, (3, 3), activation='relu', padding='same')(Input_img)
    x1 = MaxPooling2D((2, 2), padding='same')(x1)
    x2 = Conv2D(32, (3, 3), activation='relu', padding='same')(x1)
    x2 = MaxPooling2D((2, 2), padding='same')(x2)
    x3 = Conv2D(64, (3, 3), activation='relu', padding='same')(x2)
    encoded = MaxPooling2D((2, 2), padding='same')(x3)

    # decoding architecture
    x3 = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    x3 = UpSampling2D((2, 2))(x3)
    x2 = Conv2D(32, (3, 3), activation='relu', padding='same')(x3)
    x2 = UpSampling2D((2, 2))(x2)
    x1 = Conv2D(16, (3, 3), activation='relu', padding='same')(x2)
    x1 = UpSampling2D((2, 2))(x1)
    decoded = Conv2D(1, (3, 3), activation='tanh', padding='same')(x1)

    autoencoder = Model(Input_img, decoded)
    return autoencoder


def AE_model_2(model_name):
    Input_img = Input(shape=cfgs.INPUT_LAYER_DIM)

    # Encoding Architecture
    # Block 1
    x1 = Conv2D(16, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block1_conv1')(Input_img)
    x1 = BatchNormalization(name='block1_bn1')(x1)
    x1 = Activation('relu', name='block1_ac1')(x1)
    x1 = Conv2D(16, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block1_conv2')(x1)
    x1 = BatchNormalization(name='block1_bn2')(x1)
    x1 = Activation('relu', name='block1_ac2')(x1)
    x1 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block1_pool')(x1)

    # Block 2
    x2 = Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block2_conv1')(x1)
    x2 = BatchNormalization(name='block2_bn1')(x2)
    x2 = Activation('relu', name='block2_ac1')(x2)
    x2 = Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block2_conv2')(x2)
    x2 = BatchNormalization(name='block2_bn2')(x2)
    x2 = Activation('relu', name='block2_ac2')(x2)
    x2 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block2_pool')(x2)

    # Block 3
    x3 = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block3_conv1')(x2)
    x3 = BatchNormalization(name='block3_bn1')(x3)
    x3 = Activation('relu', name='block3_ac1')(x3)
    x3 = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block3_conv2')(x3)
    x3 = BatchNormalization(name='block3_bn2')(x3)
    x3 = Activation('relu', name='block3_ac2')(x3)
    x3 = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block3_conv3')(x3)
    x3 = BatchNormalization(name='block3_bn3')(x3)
    x3 = Activation('relu', name='block3_ac3')(x3)
    x3 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block3_pool')(x3)

    # Block 4
    x4 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block4_conv1')(x3)
    x4 = BatchNormalization(name='block4_bn1')(x4)
    x4 = Activation('relu', name='block4_ac1')(x4)
    x4 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block4_conv2')(x4)
    x4 = BatchNormalization(name='block4_bn2')(x4)
    x4 = Activation('relu', name='block4_ac2')(x4)
    x4 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block4_conv3')(x4)
    x4 = BatchNormalization(name='block4_bn3')(x4)
    x4 = Activation('relu', name='block4_ac3')(x4)
    x4 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block4_pool')(x4)

    # Block 5
    x5 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block5_conv1')(x4)
    x5 = BatchNormalization(name='block5_bn1')(x5)
    x5 = Activation('relu', name='block5_ac1')(x5)
    x5 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block5_conv2')(x5)
    x5 = BatchNormalization(name='block5_bn2')(x5)
    x5 = Activation('relu', name='block5_ac2')(x5)
    x5 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block5_conv3')(x5)
    x5 = BatchNormalization(name='block5_bn3')(x5)
    x5 = Activation('relu', name='block5_ac3')(x5)
    x5 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block5_pool')(x5)

    encoded = x5

    # Decoding Architecture
    # Block 6
    x6 = UpSampling2D((2, 2), name='block6_pool')(encoded)
    x6 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block6_conv1')(x6)
    x6 = BatchNormalization(name='block6_bn1')(x6)
    x6 = Activation('relu', name='block6_ac1')(x6)
    x6 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block6_conv2')(x6)
    x6 = BatchNormalization(name='block6_bn2')(x6)
    x6 = Activation('relu', name='block6_ac2')(x6)
    x6 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block6_conv3')(x6)
    x6 = BatchNormalization(name='block6_bn3')(x6)
    x6 = Activation('relu', name='block6_ac3')(x6)

    # Block 7
    x7 = UpSampling2D((2, 2), name='block7_pool')(x6)
    x7 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block7_conv1')(x7)
    x7 = BatchNormalization(name='block7_bn1')(x7)
    x7 = Activation('relu', name='block7_ac1')(x7)
    x7 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block7_conv2')(x7)
    x7 = BatchNormalization(name='block7_bn2')(x7)
    x7 = Activation('relu', name='block7_ac2')(x7)
    x7 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block7_conv3')(x7)
    x7 = BatchNormalization(name='block7_bn3')(x7)
    x7 = Activation('relu', name='block7_ac3')(x7)

    # Block 8
    x8 = UpSampling2D((2, 2), name='block8_pool')(x7)
    x8 = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block8_conv1')(x8)
    x8 = BatchNormalization(name='block8_bn1')(x8)
    x8 = Activation('relu', name='block8_ac1')(x8)
    x8 = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block8_conv2')(x8)
    x8 = BatchNormalization(name='block8_bn2')(x8)
    x8 = Activation('relu', name='block8_ac2')(x8)
    x8 = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block8_conv3')(x8)
    x8 = BatchNormalization(name='block8_bn3')(x8)
    x8 = Activation('relu', name='block8_ac3')(x8)

    # Block 9
    x9 = UpSampling2D((2, 2), name='block9_pool')(x8)
    x9 = Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block9_conv1')(x9)
    x9 = BatchNormalization(name='block9_bn1')(x9)
    x9 = Activation('relu', name='block9_ac1')(x9)
    x9 = Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block9_conv2')(x9)
    x9 = BatchNormalization(name='block9_bn2')(x9)
    x9 = Activation('relu', name='block9_ac2')(x9)

    # Block 10
    x10 = UpSampling2D((2, 2), name='block10_pool')(x9)
    x10 = Conv2D(16, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block10_conv1')(x10)
    x10 = BatchNormalization(name='block10_bn1')(x10)
    x10 = Activation('relu', name='block10_ac1')(x10)
    x10 = Conv2D(16, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block10_conv2')(x10)
    x10 = BatchNormalization(name='block10_bn2')(x10)
    x10 = Activation('relu', name='block10_ac2')(x10)

    decoded = Conv2D(1, (3, 3), activation='tanh', padding='same', name='block10_output')(x10)

    autoencoder = Model(inputs=Input_img, outputs=decoded, name=model_name)
    return autoencoder


def AE_model_3(model_name):
    Input_img = Input(shape=cfgs.INPUT_LAYER_DIM)

    # Encoding Architecture
    # Block 1
    x1 = Conv2D(16, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block1_conv1')(Input_img)
    x1 = BatchNormalization(name='block1_bn1')(x1)
    x1 = Activation('relu', name='block1_ac1')(x1)
    x1 = Conv2D(16, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block1_conv2')(x1)
    x1 = BatchNormalization(name='block1_bn2')(x1)
    x1 = Activation('relu', name='block1_ac2')(x1)
    x1 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block1_pool')(x1)

    # Block 2
    x2 = Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block2_conv1')(x1)
    x2 = BatchNormalization(name='block2_bn1')(x2)
    x2 = Activation('relu', name='block2_ac1')(x2)
    x2 = Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block2_conv2')(x2)
    x2 = BatchNormalization(name='block2_bn2')(x2)
    x2 = Activation('relu', name='block2_ac2')(x2)
    x2 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block2_pool')(x2)

    # Block 3
    x3 = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block3_conv1')(x2)
    x3 = BatchNormalization(name='block3_bn1')(x3)
    x3 = Activation('relu', name='block3_ac1')(x3)
    x3 = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block3_conv2')(x3)
    x3 = BatchNormalization(name='block3_bn2')(x3)
    x3 = Activation('relu', name='block3_ac2')(x3)
    x3 = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block3_conv3')(x3)
    x3 = BatchNormalization(name='block3_bn3')(x3)
    x3 = Activation('relu', name='block3_ac3')(x3)
    x3 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block3_pool')(x3)

    # Block 4
    x4 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block4_conv1')(x3)
    x4 = BatchNormalization(name='block4_bn1')(x4)
    x4 = Activation('relu', name='block4_ac1')(x4)
    x4 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block4_conv2')(x4)
    x4 = BatchNormalization(name='block4_bn2')(x4)
    x4 = Activation('relu', name='block4_ac2')(x4)
    x4 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block4_conv3')(x4)
    x4 = BatchNormalization(name='block4_bn3')(x4)
    x4 = Activation('relu', name='block4_ac3')(x4)
    x4 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block4_pool')(x4)

    # Block 5
    x5 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block5_conv1')(x4)
    x5 = BatchNormalization(name='block5_bn1')(x5)
    x5 = Activation('relu', name='block5_ac1')(x5)
    x5 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block5_conv2')(x5)
    x5 = BatchNormalization(name='block5_bn2')(x5)
    x5 = Activation('relu', name='block5_ac2')(x5)
    x5 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block5_conv3')(x5)
    x5 = BatchNormalization(name='block5_bn3')(x5)
    x5 = Activation('relu', name='block5_ac3')(x5)
    x5 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block5_pool')(x5)

    encoded = x5

    # Decoding Architecture
    # Block 6
    x6 = UpSampling2D((2, 2), name='block6_pool')(encoded)
    x6 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block6_conv1')(x6)
    x6 = BatchNormalization(name='block6_bn1')(x6)
    x6 = Activation('relu', name='block6_ac1')(x6)
    x6 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block6_conv2')(x6)
    x6 = BatchNormalization(name='block6_bn2')(x6)
    x6 = Activation('relu', name='block6_ac2')(x6)
    x6 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block6_conv3')(x6)
    x6 = BatchNormalization(name='block6_bn3')(x6)
    x6 = Activation('relu', name='block6_ac3')(x6)

    # Block 7
    x7 = UpSampling2D((2, 2), name='block7_pool')(x6)
    x7 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block7_conv1')(x7)
    x7 = BatchNormalization(name='block7_bn1')(x7)
    x7 = Activation('relu', name='block7_ac1')(x7)
    x7 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block7_conv2')(x7)
    x7 = BatchNormalization(name='block7_bn2')(x7)
    x7 = Activation('relu', name='block7_ac2')(x7)
    x7 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block7_conv3')(x7)
    x7 = BatchNormalization(name='block7_bn3')(x7)
    x7 = Activation('relu', name='block7_ac3')(x7)

    # Block 8
    x8 = UpSampling2D((2, 2), name='block8_pool')(x7)
    x8 = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block8_conv1')(x8)
    x8 = BatchNormalization(name='block8_bn1')(x8)
    x8 = Activation('relu', name='block8_ac1')(x8)
    x8 = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block8_conv2')(x8)
    x8 = BatchNormalization(name='block8_bn2')(x8)
    x8 = Activation('relu', name='block8_ac2')(x8)
    x8 = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block8_conv3')(x8)
    x8 = BatchNormalization(name='block8_bn3')(x8)
    x8 = Activation('relu', name='block8_ac3')(x8)

    # Block 9
    x9 = UpSampling2D((2, 2), name='block9_pool')(x8)
    x9 = Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block9_conv1')(x9)
    x9 = BatchNormalization(name='block9_bn1')(x9)
    x9 = Activation('relu', name='block9_ac1')(x9)
    x9 = Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block9_conv2')(x9)
    x9 = BatchNormalization(name='block9_bn2')(x9)
    x9 = Activation('relu', name='block9_ac2')(x9)

    # Block 10
    x10 = UpSampling2D((2, 2), name='block10_pool')(x9)
    x10 = Conv2D(16, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block10_conv1')(x10)
    x10 = BatchNormalization(name='block10_bn1')(x10)
    x10 = Activation('relu', name='block10_ac1')(x10)
    x10 = Conv2D(16, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block10_conv2')(x10)
    x10 = BatchNormalization(name='block10_bn2')(x10)
    x10 = Activation('relu', name='block10_ac2')(x10)

    decoded = Conv2D(1, (3, 3), activation='tanh', padding='same', name='block10_output')(x10)

    autoencoder = Model(inputs=Input_img, outputs=decoded, name=model_name)
    return autoencoder


# Reference: U-Net
def AE_model_4(model_name):
    Input_img = Input(shape=cfgs.INPUT_LAYER_DIM)

    # Setup
    activation = 'swish'

    # Encoding Architecture
    # Block 1
    x1 = Conv2D(16, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block1_conv1')(Input_img)
    x1 = BatchNormalization(name='block1_bn1')(x1)
    x1 = Activation(activation, name='block1_ac1')(x1)
    x1 = Conv2D(16, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block1_conv2')(x1)
    x1 = BatchNormalization(name='block1_bn2')(x1)
    x1 = Activation(activation, name='block1_ac2')(x1)
    x1 = Conv2D(16, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block1_conv3')(x1)
    x1 = BatchNormalization(name='block1_bn3')(x1)
    x1 = Activation(activation, name='block1_ac3')(x1)
    x1d = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block1_pool')(x1)

    # Block 2
    x2 = Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block2_conv1')(x1d)
    x2 = BatchNormalization(name='block2_bn1')(x2)
    x2 = Activation(activation, name='block2_ac1')(x2)
    x2 = Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block2_conv2')(x2)
    x2 = BatchNormalization(name='block2_bn2')(x2)
    x2 = Activation(activation, name='block2_ac2')(x2)
    x2 = Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block2_conv3')(x2)
    x2 = BatchNormalization(name='block2_bn3')(x2)
    x2 = Activation(activation, name='block2_ac3')(x2)
    x2d = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block2_pool')(x2)

    # Block 3
    x3 = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block3_conv1')(x2d)
    x3 = BatchNormalization(name='block3_bn1')(x3)
    x3 = Activation(activation, name='block3_ac1')(x3)
    x3 = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block3_conv2')(x3)
    x3 = BatchNormalization(name='block3_bn2')(x3)
    x3 = Activation(activation, name='block3_ac2')(x3)
    x3 = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block3_conv3')(x3)
    x3 = BatchNormalization(name='block3_bn3')(x3)
    x3 = Activation(activation, name='block3_ac3')(x3)
    x3d = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block3_pool')(x3)

    # Block 4
    x4 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block4_conv1')(x3d)
    x4 = BatchNormalization(name='block4_bn1')(x4)
    x4 = Activation(activation, name='block4_ac1')(x4)
    x4 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block4_conv2')(x4)
    x4 = BatchNormalization(name='block4_bn2')(x4)
    x4 = Activation(activation, name='block4_ac2')(x4)
    x4 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block4_conv3')(x4)
    x4 = BatchNormalization(name='block4_bn3')(x4)
    x4 = Activation(activation, name='block4_ac3')(x4)
    x4d = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block4_pool')(x4)

    # Block 5
    x5 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block5_conv1')(x4d)
    x5 = BatchNormalization(name='block5_bn1')(x5)
    x5 = Activation(activation, name='block5_ac1')(x5)
    x5 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block5_conv2')(x5)
    x5 = BatchNormalization(name='block5_bn2')(x5)
    x5 = Activation(activation, name='block5_ac2')(x5)
    x5 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block5_conv3')(x5)
    x5 = BatchNormalization(name='block5_bn3')(x5)
    x5 = Activation(activation, name='block5_ac3')(x5)
    #x5 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block5_pool')(x5)

    encoded = x5

    # Decoding Architecture
    # Block 6
    #x6 = UpSampling2D((2, 2), name='block6_pool')(encoded)
    #x6 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block6_conv1')(x6)
    #x6 = BatchNormalization(name='block6_bn1')(x6)
    #x6 = Activation(activation, name='block6_ac1')(x6)
    #x6 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block6_conv2')(x6)
    #x6 = BatchNormalization(name='block6_bn2')(x6)
    #x6 = Activation(activation, name='block6_ac2')(x6)
    #x6 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block6_conv3')(x6)
    #x6 = BatchNormalization(name='block6_bn3')(x6)
    #x6 = Activation(activation, name='block6_ac3')(x6)

    # Block 7
    x7 = UpSampling2D((2, 2), name='block7_pool')(encoded)
    x7 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block7_conv1')(x7)
    x7 = BatchNormalization(name='block7_bn1')(x7)
    x7 = Activation(activation, name='block7_ac1')(x7)
    x7 = Concatenate(axis=3)([x4, x7])
    x7 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block7_conv2')(x7)
    x7 = BatchNormalization(name='block7_bn2')(x7)
    x7 = Activation(activation, name='block7_ac2')(x7)
    x7 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block7_conv3')(x7)
    x7 = BatchNormalization(name='block7_bn3')(x7)
    x7 = Activation(activation, name='block7_ac3')(x7)

    # Block 8
    x8 = UpSampling2D((2, 2), name='block8_pool')(x7)
    x8 = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block8_conv1')(x8)
    x8 = BatchNormalization(name='block8_bn1')(x8)
    x8 = Activation(activation, name='block8_ac1')(x8)
    x8 = Concatenate(axis=3)([x3, x8])
    x8 = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block8_conv2')(x8)
    x8 = BatchNormalization(name='block8_bn2')(x8)
    x8 = Activation(activation, name='block8_ac2')(x8)
    x8 = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block8_conv3')(x8)
    x8 = BatchNormalization(name='block8_bn3')(x8)
    x8 = Activation(activation, name='block8_ac3')(x8)

    # Block 9
    x9 = UpSampling2D((2, 2), name='block9_pool')(x8)
    x9 = Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block9_conv1')(x9)
    x9 = BatchNormalization(name='block9_bn1')(x9)
    x9 = Activation(activation, name='block9_ac1')(x9)
    x9 = Concatenate(axis=3)([x2, x9])
    x9 = Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block9_conv2')(x9)
    x9 = BatchNormalization(name='block9_bn2')(x9)
    x9 = Activation(activation, name='block9_ac2')(x9)
    x9 = Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block9_conv3')(x9)
    x9 = BatchNormalization(name='block9_bn3')(x9)
    x9 = Activation(activation, name='block9_ac3')(x9)

    # Block 10
    x10 = UpSampling2D((2, 2), name='block10_pool')(x9)
    x10 = Conv2D(16, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block10_conv1')(x10)
    x10 = BatchNormalization(name='block10_bn1')(x10)
    x10 = Activation(activation, name='block10_ac1')(x10)
    x10 = Concatenate(axis=3)([x1, x10])
    x10 = Conv2D(16, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block10_conv2')(x10)
    x10 = BatchNormalization(name='block10_bn2')(x10)
    x10 = Activation(activation, name='block10_ac2')(x10)
    x10 = Conv2D(16, (3, 3), padding='same', kernel_regularizer=regularizers.l2(cfgs.L2_REGULAR), kernel_initializer='he_normal', name='block10_conv3')(x10)
    x10 = BatchNormalization(name='block10_bn3')(x10)
    x10 = Activation(activation, name='block10_ac3')(x10)

    decoded = Conv2D(1, (3, 3), activation='tanh', padding='same', kernel_initializer='glorot_normal', name='block10_output')(x10)

    autoencoder = Model(inputs=Input_img, outputs=decoded, name=model_name)
    return autoencoder


# Reference: U-Net
def AE_model_4_1(model_name):
    Input_img = Input(shape=cfgs.INPUT_LAYER_DIM)

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

    # Block 7
    x7 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal', name='block7_convT1')(encoded)
    #x7 = UpSampling2D((2, 2), name='block7_pool')(encoded)
    #x7 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', name='block7_conv1')(x7)
    x7 = Activation(activation, name='block7_ac1')(x7)
    x7 = Concatenate(axis=3)([x4, x7])
    x7 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', name='block7_conv2')(x7)
    x7 = Activation(activation, name='block7_ac2')(x7)
    x7 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', name='block7_conv3')(x7)
    x7 = Activation(activation, name='block7_ac3')(x7)

    # Block 8
    x8 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal', name='block8_convT1')(x7)
    #x8 = UpSampling2D((2, 2), name='block8_pool')(x7)
    #x8 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', name='block8_conv1')(x8)
    x8 = Activation(activation, name='block8_ac1')(x8)
    x8 = Concatenate(axis=3)([x3, x8])
    x8 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', name='block8_conv2')(x8)
    x8 = Activation(activation, name='block8_ac2')(x8)
    x8 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', name='block8_conv3')(x8)
    x8 = Activation(activation, name='block8_ac3')(x8)

    # Block 9
    x9 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal', name='block9_convT1')(x8)
    #x9 = UpSampling2D((2, 2), name='block9_pool')(x8)
    #x9 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal', name='block9_conv1')(x9)
    x9 = Activation(activation, name='block9_ac1')(x9)
    x9 = Concatenate(axis=3)([x2, x9])
    x9 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal', name='block9_conv2')(x9)
    x9 = Activation(activation, name='block9_ac2')(x9)
    x9 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal', name='block9_conv3')(x9)
    x9 = Activation(activation, name='block9_ac3')(x9)

    # Block 10
    x10 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal', name='block10_convT1')(x9)
    #x10 = UpSampling2D((2, 2), name='block10_pool')(x9)
    #x10 = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal', name='block10_conv1')(x10)
    x10 = Activation(activation, name='block10_ac1')(x10)
    x10 = Concatenate(axis=3)([x1, x10])
    x10 = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal', name='block10_conv2')(x10)
    x10 = Activation(activation, name='block10_ac2')(x10)
    x10 = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal', name='block10_conv3')(x10)
    x10 = Activation(activation, name='block10_ac3')(x10)
    #x10 = Activation(activation='tanh', name='block10_ac3')(x10)

    decoded = Conv2D(1, (3, 3), activation='tanh', padding='same', kernel_initializer='glorot_normal', name='block10_output')(x10)
    #decoded = Lambda(lambda x: mean(x, axis=3)[:, :, :, None])(x10)

    autoencoder = Model(inputs=Input_img, outputs=decoded, name=model_name)
    return autoencoder
