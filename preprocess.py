import cv2
import numpy as np
import parameters


def preprocess(image):
    # crop image
    ret = image[parameters.IMAGE_VERTICAL_SLICE, parameters.IMAGE_HORIZONTAL_SLICE, :]
    # convert to yuv
    ret = cv2.cvtColor(ret, cv2.COLOR_BGR2YUV)
    # resize to a desired sample size
    ret = cv2.resize(ret, parameters.SAMPLE_SHAPE[1::-1], interpolation=cv2.INTER_LINEAR)
    # equalize luma channel
    ret[:, :, 0] = cv2.equalizeHist(ret[:, :, 0])

    if parameters.SAMPLE_SHAPE[-1] == 1:
        ret = ret[:, :, 0]
    else:
        # convert to BGR
        ret = cv2.cvtColor(ret, cv2.COLOR_YUV2BGR)
    # return as float32, with 0-1 range
    return ret.astype(np.float32) / 256


if __name__ == '__main__':
    test_image = 'images/track0_hard_1/IMG/center_2017_03_20_15_52_25_613.jpg'
    img = cv2.imread(test_image)
    print(img.shape)
    cv2.namedWindow('sample', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('sample', 640, 480)
    cv2.namedWindow('prepro', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('prepro', 640, 480)
    cv2.imshow('sample', img)
    cv2.imshow('prepro', preprocess(img))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
