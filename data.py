import os
import pandas as pd
from sklearn.model_selection import train_test_split
import parameters
import numpy as np
from preprocess import preprocess
import cv2


def database():
    """
    Builds pandas dataframe with training data.
    """
    data_root = 'images'
    log_name = 'driving_log.csv'
    runs = list(sorted(filter(lambda pt: os.path.isdir(os.path.join(data_root, pt)) and
                                         os.path.exists(os.path.join(data_root, pt, log_name)), os.listdir(data_root))))
    dataframe = pd.DataFrame()
    column_names = ['center', 'left', 'right', 'angle', 'throttle', 'break', 'speed']

    for run in runs:
        rd = pd.read_csv(os.path.join(data_root, run, log_name), header=None, names=column_names)
        rd['left'] = rd['left'].apply(lambda x: os.path.join(data_root, run, x))
        rd['center'] = rd['center'].apply(lambda x: os.path.join(data_root, run, x))
        rd['right'] = rd['right'].apply(lambda x: os.path.join(data_root, run, x))
        dataframe = dataframe.append(rd)

    return dataframe


def validation_split(df):
    df_train, df_validation = train_test_split(df, test_size=parameters.VALIDATION_FRACTION)
    return df_train, df_validation


def load_data(df):
    x_data = np.zeros((len(df), *parameters.SAMPLE_SHAPE), dtype=np.float32)
    y_data = np.zeros(len(df), np.float32)
    index = 0
    for _, item in df.iterrows():
        x_data[index, :, :, :], y_data[index] = np.atleast_3d(preprocess(cv2.imread(item['center']))), item['angle']
        index += 1
    return x_data, y_data


def augment_data(x, y):

    for idx in range(len(y)):
        # random horizontal flip
        if np.random.rand() > 0.5:
            x[idx, :, :, :] = np.atleast_3d(cv2.flip(np.squeeze(x[idx, :, :, :]), 1))
            y[idx] = -y[idx]
        # random inverse
        if np.random.rand() > 0.5:
            x[idx, :, :, :] = 1 - x[idx, :, :, :]

    return x, y


def generator(df, augment=True):
    """
    yields batches sampled randomly
    """
    while 1:
        batch = df.sample(n=parameters.GENERATOR_BATCH_SIZE)
        if augment:
            yield augment_data(*load_data(batch))
        else:
            yield load_data(batch)
