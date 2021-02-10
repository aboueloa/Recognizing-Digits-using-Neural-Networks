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

# different numbers of convolutional layers
index = [1, 2, 3]
#=========================================================================
def cnn_classifier():
    for i in range(len(index)):

      classifier = Sequential()

      classifier.add(Conv2D(16, (3, 3), input_shape = (28, 28, 1)))
      classifier.add(Activation('relu'))
      classifier.add(MaxPooling2D(pool_size = (2, 2)))

      if i>0:
        classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))

      if i>1:
        classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))

      classifier.add(Flatten())

      classifier.add(Dense(units = 20, kernel_initializer = 'uniform'))
      classifier.add(Activation('relu'))

      classifier.add(Dense(units = 10, kernel_initializer = 'uniform'))
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
conv = ["1 conv layer", "2 conv layers", "3 conv layers"]
# plotting average of each size
barplot_acc_loss(conv, history)
# plotting the progression of each size during epochs
plot_train_test(conv, history, best=1)
# results
show_results(conv, history)
#===========================================================================
