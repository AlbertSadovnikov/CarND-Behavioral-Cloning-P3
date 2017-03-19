import pandas as pd
import numpy as np
import cv2
import os
from preprocess import preprocess_center
from model import nvidia_model
from keras.optimizers import Adam
from scipy import stats
import matplotlib.pyplot as plt
import sys


# check data/
data_root = 'data'
log_name = 'driving_log.csv'
runs = list(sorted(filter(lambda pt: os.path.isdir(os.path.join(data_root, pt)) and
                                     os.path.exists(os.path.join(data_root, pt, log_name)), os.listdir(data_root))))

df = []
column_names = ['center', 'left', 'right', 'angle', 'throttle', 'break', 'speed']
for run in runs:
    rd = pd.read_csv(os.path.join(data_root, run, log_name), header=None, names=column_names)
    rd['left'] = rd['left'].apply(lambda x: os.path.join(data_root, run, x))
    rd['center'] = rd['center'].apply(lambda x: os.  path.join(data_root, run, x))
    rd['right'] = rd['right'].apply(lambda x: os.path.join(data_root, run, x))
    df.append(rd)
    print('Loaded %s with %d samples.' % (run, len(rd)))

# build up the training data: X_full (list of mat files) and y_full (angles)
# add flipped variants, shuffle and split into training and validation
#
# img = cv2.imread(df[0]['center'][100])
# plt.figure()
# plt.imshow(img)
# rimg = cv2.flip(img, 1)
# plt.figure()
# plt.imshow(rimg)
# plt.show()
# import sys
# sys.exit()


input_shape = (64, 128, 3)
X_train = []
X_train_left = []
X_train_right = []
y_train = []

for rd in df:
    X_train.extend([preprocess_center(cv2.imread(row[1]),
                                      target_size=input_shape[:2]) for row in rd.itertuples()])
    y_train.extend([row[4] for row in rd.itertuples()])
    # flipped
    X_train.extend([preprocess_center(cv2.flip(cv2.imread(row[1]), 1),
                                      target_size=input_shape[:2]) for row in rd.itertuples()])
    y_train.extend([-row[4] for row in rd.itertuples()])

    # add right
    X_train.extend([preprocess_center(cv2.imread(row[2]),
                                      target_size=input_shape[:2]) for row in rd.itertuples()])

    y_train.extend([row[4] + 0.1 for row in rd.itertuples()])

    # add right flipped
    X_train.extend([preprocess_center(cv2.imread(row[2]),
                                      target_size=input_shape[:2]) for row in rd.itertuples()])

    y_train.extend([-row[4] - 0.1 for row in rd.itertuples()])

    # add right
    X_train.extend([preprocess_center(cv2.imread(row[3]),
                                      target_size=input_shape[:2]) for row in rd.itertuples()])

    y_train.extend([row[4] - 0.1 for row in rd.itertuples()])

    # add right flipped
    X_train.extend([preprocess_center(cv2.imread(row[3]),
                                      target_size=input_shape[:2]) for row in rd.itertuples()])

    y_train.extend([-row[4] + 0.1 for row in rd.itertuples()])

X_train = np.array(X_train)
y_train = np.array(y_train)

model = nvidia_model(input_shape)
model.summary()
model.compile(loss='mse', optimizer=Adam(lr=0.0001), metrics=['mse', 'accuracy'])
model.load_weights('model.h5')
model.fit(X_train, y_train, validation_split=0.1, shuffle=True, epochs=25, batch_size=64)
model.save('model.h5')











