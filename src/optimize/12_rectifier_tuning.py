#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '..')

# Importing Keras libraries
# plotting & image preprocessing functions
from tools.image_preprocessing import *
from tools.plotting import *
from tools.history import *

#==============================================================================
# Scaled & well prepared Data
(train_x, train_y) ,(test_x, test_y) = preparing_data()

# different leaky_relu slopes
index = [0, 0.05, 0.1, 0.5, 1]

#=========================================================================
def cnn_classifier():
    for i in index:
      classifier = Sequential()

      classifier.add(Conv2D(64, (3, 3), input_shape = (28, 28, 1)))
      BatchNormalization(axis=-1)
      classifier.add(LeakyReLU(i))
      BatchNormalization(axis=-1)

      classifier.add(MaxPooling2D(pool_size = (2, 2)))
      BatchNormalization(axis=-1)
      classifier.add(Dropout(0.2))
      BatchNormalization(axis=-1)

      classifier.add(Conv2D(128, (3, 3)))
      BatchNormalization(axis=-1)
      classifier.add(LeakyReLU(i))
      BatchNormalization(axis=-1)
      classifier.add(MaxPooling2D(pool_size = (2, 2)))
      classifier.add(Dropout(0.2))

      classifier.add(Flatten())
      classifier.add(BatchNormalization())
      classifier.add(Dense(units = 128, kernel_initializer = 'uniform'))
      classifier.add(BatchNormalization())
      classifier.add(Activation('relu'))
      classifier.add(Dropout(0.2))
      classifier.add(BatchNormalization())
      classifier.add(Dense(units = 10, kernel_initializer = 'uniform'))
      classifier.add(BatchNormalization())
      classifier.add(Activation('softmax'))

      classifier.compile(optimizer = Adam(learning_rate=0.01),
                         loss = 'categorical_crossentropy',
                         metrics = ['accuracy'])
      yield classifier

#==============================================================================
# Get classifiers
classifiers = cnn_classifier()

#=========================================================================
# Get histories of our models
history = get_history_with_test(len(index), classifiers,
                                (train_x, train_y),
                                (test_x, test_y),
                                256, 25)

#=========================================================================
index = ["Slope = 0", "Slope = 0.05", "Slope = 0.1", "Slope = 0.5", "Slope = 1"]
# plotting average of each size
barplot_acc_loss(index, history)
# plotting the progression of each size during epochs
plot_train_test(index, history, best=1)
# results
show_results(index, history)
#===========================================================================
