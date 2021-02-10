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

# different optimizers
index = ['RMSprop', 'Nadam', 'GD', 'Adadelta', 'Adagrad', 'Adam']

#==============================================================================
# building our classifiers

def optimizer_cnn():
    for opt in [RMSprop(learning_rate=0.01), Nadam(learning_rate=0.01),
                SGD(learning_rate=0.1), Adadelta(learning_rate=1),
                Adagrad(learning_rate=0.1), Adam(learning_rate=0.01)]:
        classifier = Sequential()
        classifier.add(Conv2D(32, (3, 3), input_shape = (28, 28, 1)))
        classifier.add(Activation('relu'))

        classifier.add(AveragePooling2D(pool_size = (2, 2)))

        classifier.add(Flatten())
        classifier.add(Dense(units = 20, kernel_initializer = 'uniform'))
        classifier.add(Activation('relu'))

        classifier.add(Dense(units = 10, kernel_initializer = 'uniform'))
        classifier.add(Activation('softmax'))

        classifier.compile(optimizer = opt, loss = 'categorical_crossentropy',
                           metrics = ['accuracy'])

        yield classifier

#==============================================================================
# Get classifiers
classifiers = optimizer_cnn()

#=========================================================================
# Get histories of our models
history = get_history(len(index), classifiers, train_x, train_y, 512, 25)


# plotting average of each size
barplot_acc_loss(index, history)
# plotting the progression of each size during epochs
plot_train_test(index, history, best=5)
# results
show_results(index, history)
#==============================================================================
