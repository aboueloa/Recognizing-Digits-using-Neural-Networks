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

# different sizes
index = [3, 5, 7]

#==============================================================================
# building our classifiers
def size_cnn():
    for i in index:
        classifier = Sequential()
        classifier.add(Conv2D(32, (i, i), input_shape = (28, 28, 1)))
        classifier.add(Activation('relu'))

        classifier.add(AveragePooling2D(pool_size = (2, 2)))

        classifier.add(Flatten())
        classifier.add(Dense(units = 20, kernel_initializer = 'uniform'))
        classifier.add(Activation('relu'))

        classifier.add(Dense(units = 10, kernel_initializer = 'uniform'))
        classifier.add(Activation('softmax'))

        classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        yield classifier

#==============================================================================
# Get classifiers
classifiers = size_cnn()

# Get histories of our models
history = get_history(len(index), classifiers, train_x, train_y, 32, 25)

sizes = ['3x3', '5x5', '7x7']
# plotting average of each size
barplot_acc_loss(sizes, history)
# plotting the progression of each size during epochs
plot_train_test(sizes, history, best=0)
# results
show_results(sizes, history)
#==============================================================================
