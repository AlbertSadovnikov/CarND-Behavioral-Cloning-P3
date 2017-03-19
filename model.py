from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers.core import Dense, Flatten
from keras.layers import BatchNormalization


def nvidia_model(input_shape):
    model = Sequential()
    model.add(BatchNormalization(epsilon=0.001, axis=3, input_shape=input_shape))
    model.add(Conv2D(24, (5, 5), padding='valid', activation='relu', strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), padding='valid', activation='relu', strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), padding='valid', activation='relu', strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), padding='valid', activation='relu', strides=(1, 1)))
    model.add(Conv2D(64, (3, 3), padding='valid', activation='relu', strides=(1, 1)))
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='tanh'))
    return model


