from keras.layers import (Activation, BatchNormalization,
                          Conv2D, Dense, Input, Flatten, MaxPooling2D, UpSampling2D)
from keras.models import Model


def NN_model_1():
    X_input = Input((51, 51, 3))
    # stage 1
    X = Conv2D(4, (3, 3), strides=(1, 1),
               name="conv1", padding="same")(X_input)
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


def AEN_model_1():
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
