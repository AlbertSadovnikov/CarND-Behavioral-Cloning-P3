# Behavioral Cloning

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[model]: ./report/model.png "Model Visualization"
[angle_dist]: ./report/data_dist.png "Angle distribution"
[sample]: ./report/original_sample.jpg "Original sample"
[preprocessed]: ./report/preprocessed_sample.jpg "Preprocessed sample"
[mirrored]: ./report/mirrored_sample.jpg "Mirrored sample"
[inverted]: ./report/inv_preprocessed_sample.jpg "Inverted sample"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

*1. Submission includes all required files and can be used to run the simulator in autonomous mode*

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* nvidia.py containing a model design
* video.mp4 containing driving recording on both test tracks
* writeup_report.md summarizing the results

*2. Submission includes functional code*

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python3 drive.py model.h5
```

*3. Submission code is usable and readable*

The model.py and included files contain the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. In most places code is self-explanatory.

### Model Architecture and Training Strategy

*1. An appropriate model architecture has been employed*

After a brief research I have found a [model from Nvidia](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) for exactly the same problem.

I have reworked it a little:
* Added batch normalization.
* Reduced input dimensions to grayscale.
* Reduced the sizes of fully-connected layers.
* Added maxpooling layer.
* Added dropout layers.

![Keras model visualization][model]


*2. Attempts to reduce overfitting in the model*

The model contains dropout layers in order to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting (see data splitting in model.py). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

*3. Model parameter tuning*

The model used an adam optimizer, so the learning rate was not tuned manually.

*4. Appropriate training data*

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. Also I've recorded a few laps only turns (trying not to save on the straight segments).

Actual training data had 36863 samples.

On the figure it can be seen that the actual data tends to have more left turns and a lot of 0 angles. So, the mirroring was needed.
![Angles distribution][angle_dist]

*5. Preprocessing*

Original images were cropped, converted to grayscale and had histgram equalization performed, to account for difference in illumination.

Original sample.

![original sample][sample]

Preprocessed sample.

![preprocessed sample][preprocessed]

*6. Augmentation*

For the data augmentation, I have used mirroring (also taking negative steering).

![Mirrored sample][mirrored]

And inverting the grayscale image, to help the network learning the edges.

![Inverted sample][inverted]

*7. Solution Design Approach Steps*

I took the model from Nvidia and updated it along the way.
Actual iterations were (in order of appearance):

1. Basic driving data (simple driving on both tracks).
2. Complex driving data (recovering from the track sides).
3. Data augmentation (mirroring).
4. Preprocessing (cropping, converting to grayscale and histogram equalization).
5. Turns only (driving around but recording only turns).
6. Data augmentation (image inversion, to support detecting lane edges).
7. Model complexity reduction, dropouts.
