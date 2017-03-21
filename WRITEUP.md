#**Behavioral Cloning**

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* The training dataset is excluded from the github, because of the size, there are appx 30K images.
* parameters.py Contains misc param definitions.
* nvidia.py Contains implementation of the original Nvidia model for predicting the steering angle. Also a modified version (for grayscale inputs) is implemented there.
* data.py Contains data manipulations.
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python3 drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed
This is the model I've used for the task. It resembles the original Nvidia one.
```python3
m = Sequential()
m.add(BatchNormalization(epsilon=0.001, input_shape=(70, 200, 1)))
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
```
Compared to the original one - it has less layers and less tunable parameters:
0.5M << 2.5M


####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 11).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

####5. Data preprocessing.
Images were converted to grayscale, a histogram equalization was used to take care of different lighting conditions.


####6. Data augmentation.
Two augmentation tricks were used - horizontal flip (also change steering angle sign).
Also, since it was important to find lane lines correctly, for that image inverse was used (1.0 - image).


####7. Training procedure.
Original dataset was split into training and validation.
The data was loaded and sampled randomly in batches.
Network converged to a usable state relatively fast <5 minutes on GTX680.

####8. Dataset used.
~60% are free runs, ~30% only (hard) turns, ~10% recovery.

####9. Video.
Video contains 2 rounds on both tracks, at 30mph (risky variant). For safety reasons the speed can be reduced:)
