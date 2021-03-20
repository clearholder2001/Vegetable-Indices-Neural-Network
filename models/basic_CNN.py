from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Dense, Flatten, Input, MaxPooling2D)
from tensorflow.keras.models import Model


def basic_CNN():
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
