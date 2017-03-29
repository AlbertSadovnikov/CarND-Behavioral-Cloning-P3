from keras.utils import plot_model
import nvidia
import data
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import preprocess
import cv2
import numpy as np

model = nvidia.model_x()
model.summary()
model.load_weights('model.h5')
plot_model(model, show_shapes=True, show_layer_names=True, to_file='report/model.png')


# data distribution
db = data.database()
d_plot = sns.distplot(db['angle'])
d_plot.get_figure().savefig('report/data_dist.png')

print(len(db))

test_image = 'images/run4/IMG/center_2017_03_19_22_39_31_990.jpg'
img = cv2.imread(test_image)
p_img = preprocess(img)
mp_img = cv2.flip(p_img, 1)

cv2.imwrite('report/original_sample.jpg', img)
cv2.imwrite('report/preprocessed_sample.jpg', np.uint8(p_img * 255))
cv2.imwrite('report/mirrored_sample.jpg', np.uint8(mp_img * 255))
cv2.imwrite('report/inv_preprocessed_sample.jpg', np.uint8((1 - p_img) * 255))
cv2.imwrite('report/inv_mirrored_sample.jpg', np.uint8((1 - mp_img) * 255))









