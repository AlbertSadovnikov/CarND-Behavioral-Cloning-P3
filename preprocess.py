import cv2
import numpy as np


def preprocess_center(img, target_size=(64, 128), top_chop=64):
    img = cv2.resize(img[top_chop:, :, :], (target_size[1], target_size[0]))
    #img_float = np.zeros((*target_size, 3), dtype=np.float32)
    #cv2.normalize(img, img_float, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32FC3)
    return img
