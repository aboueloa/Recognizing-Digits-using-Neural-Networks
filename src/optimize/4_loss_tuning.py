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

# different loss functions
index = ['kullback_leibler_divergence', 'categorical_crossentropy', 'sparse_categorical_crossentropy']

#=========================================================================
def loss_cnn():
    for loss in index:
        classifier = Sequential()
        classifier.add(Conv2D(128, (3, 3), input_shape = (28, 28, 1)))
        classifier.add(Activation('relu'))

        classifier.add(AveragePooling2D(pool_size = (2, 2)))

        classifier.add(Flatten())
        classifier.add(Dense(units = 20, kernel_initializer = 'uniform'))
        classifier.add(Activation('relu'))

        classifier.add(Dense(units = 10, kernel_initializer = 'uniform'))
        classifier.add(Activation('softmax'))

        classifier.compile(optimizer = Adam(learning_rate=0.01), loss = loss,
                           metrics = ['accuracy'])

        yield classifier

#==============================================================================
# Get classifiers
classifiers = loss_cnn()

# Get histories of our models
i = 0
history = [0] * 3
for classifier in classifiers:
    # when loss == 'sparse_categorical_crossentropy'
    if i == 2:
        train_y = np.argmax(train_y,axis = 1)
        test_y = np.argmax(test_y,axis = 1)
    history[i] = classifier.fit(train_x, train_y,
                                validation_data = (test_x, test_y),
                                batch_size=512, epochs=25)
    i += 1
#==============================================================================
# plotting average of each size
barplot_acc_loss(index, history)
# plotting the progression of each size during epochs
plot_train_test(index, history, best=1)
# results
show_results(index, history)
#==============================================================================
