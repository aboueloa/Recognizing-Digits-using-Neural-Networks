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

# different numbers of nodes
index = [32, 64, 128, 256, 512, 1605, 3210]
#=========================================================================

def cnn_classifier():
    for i in index:
      classifier = Sequential()

      classifier.add(Conv2D(64, (3, 3), input_shape = (28, 28, 1)))
      classifier.add(Activation('relu'))
      classifier.add(MaxPooling2D(pool_size = (2, 2)))

      classifier.add(Conv2D(128, (3, 3), activation = 'relu'))
      classifier.add(MaxPooling2D(pool_size = (2, 2)))

      classifier.add(Flatten())

      classifier.add(Dense(units = i, kernel_initializer = 'uniform'))
      classifier.add(Activation('relu'))


      classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'softmax'))

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
index = ["32 units", "64 units", "128 units", "256 units", "512 units", "1605 units", "3210 units"]
# plotting average of each size
barplot_acc_loss(index, history)
# plotting the progression of each size during epochs
plot_train_test(index, history, best=2)
# results
show_results(index, history)
#===========================================================================
