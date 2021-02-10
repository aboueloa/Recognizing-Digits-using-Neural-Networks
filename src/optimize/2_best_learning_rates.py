#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '..')

# Importing Keras libraries
# plotting & image preprocessing functions
from tools.image_preprocessing import *
from tools.plotting import *
from tools.history import *
from tools.learning_rate import LRFinder

#==============================================================================
# Scaled & well prepared Data
(train_x, train_y) ,(test_x, test_y) = preparing_data()

# different optimizers
index = ['RMSprop', 'Nadam', 'GD', 'Adadelta', 'Adagrad', 'Adam']

#==============================================================================
# building our classifiers
def optimizer_rate_cnn():
    for opt in [RMSprop(), Nadam(), SGD(), Adadelta(), Adagrad(), Adam()]:
        classifier = Sequential()
        classifier.add(Conv2D(32, (3, 3), input_shape = (28, 28, 1)))
        classifier.add(Activation('relu'))

        classifier.add(AveragePooling2D(pool_size = (2, 2)))

        classifier.add(Flatten())
        classifier.add(Dense(units = 20, kernel_initializer = 'uniform'))
        classifier.add(Activation('relu'))

        classifier.add(Dense(units = 10, kernel_initializer = 'uniform'))
        classifier.add(Activation('softmax'))

        classifier.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])

        yield classifier

#==============================================================================
# Get classifiers
classifiers = optimizer_rate_cnn()


# plotting learning rates
i = 0
for classifier in classifiers:
    lr_finder = LRFinder(classifier)
    lr_finder.find(train_x, train_y, start_lr=0.0000001, end_lr=100,
                   batch_size=512, epochs=25)
    lr_finder.plot_loss(n_skip_beginning=20, n_skip_end=5)
    i += 1
plt.legend(index, loc='upper right')
plt.show()
#==============================================================================
