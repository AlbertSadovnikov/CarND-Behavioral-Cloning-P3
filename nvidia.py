from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import PReLU
import parameters


def model():
    """
    NVIDIA Model
    https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    """
    m = Sequential()
    m.add(BatchNormalization(epsilon=0.001, input_shape=parameters.SAMPLE_SHAPE))
    m.add(Conv2D(24, (5, 5), padding='valid', activation='relu', strides=(2, 2)))
    m.add(Conv2D(36, (5, 5), padding='valid', activation='relu', strides=(2, 2)))
    m.add(Conv2D(48, (5, 5), padding='valid', activation='relu', strides=(2, 2)))
    m.add(Conv2D(64, (3, 3), padding='valid', activation='relu', strides=(1, 1)))
    m.add(Conv2D(64, (3, 3), padding='valid', activation='relu', strides=(1, 1)))
    m.add(Flatten())
    m.add(Dense(1164, activation='relu'))
    m.add(Dense(100, activation='relu'))
    m.add(Dense(50, activation='relu'))
    m.add(Dense(10, activation='relu'))
    m.add(Dense(1, activation='tanh'))
    return m


def model_x():
    """
    Slightly modified NVIDIA model
    """
    m = Sequential()
    m.add(BatchNormalization(epsilon=0.001, input_shape=parameters.SAMPLE_SHAPE))
    m.add(Conv2D(24, (5, 5), padding='valid', activation='relu', strides=(2, 2)))
    m.add(Conv2D(36, (5, 5), padding='valid', activation='relu', strides=(2, 2)))
    m.add(Conv2D(48, (5, 5), padding='valid', activation='relu', strides=(2, 2)))
    m.add(Dropout(0.25))
    m.add(Conv2D(64, (3, 3), padding='valid', activation='relu', strides=(1, 1)))
    m.add(Conv2D(64, (3, 3), padding='valid', activation='relu', strides=(1, 1)))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Dropout(0.25))
    m.add(Flatten())
    m.add(Dense(512, activation='relu'))
    m.add(Dense(256, activation='relu'))
    m.add(Dropout(0.5))
    m.add(Dense(1, activation='tanh'))
    return m

