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

#============================================================================
# Data augmentation
def augmentation(shear, wshift, rot, zoom):
    test_datagen = ImageDataGenerator()
    train_datagen = ImageDataGenerator(
                                    shear_range = shear,
                                    width_shift_range=wshift,
                                    rotation_range=rot,
                                    zoom_range = zoom,
                                    fill_mode="nearest"
                                    )
    training_set = train_datagen.flow(train_x, train_y, batch_size = 256)
    test_set = test_datagen.flow(test_x, test_y, batch_size = 256)
    return (training_set, test_set)

(training_set, test_set) = augmentation(shear=0.1, wshift=0.02, rot=8, zoom=0.08)

#============================================================================
# Show transformed images.
import matplotlib.pyplot as plt
for X_batch, y_batch in training_set:
	# create a grid of 3x3 images
	for i in range(0, 9):
		plt.subplot(331 + i)
		plt.imshow(X_batch[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
	plt.show()
	break

#===========================================================================
def final_cnn():
  classifier = Sequential()

  classifier.add(Conv2D(64, (3, 3), input_shape = (28, 28, 1)))
  BatchNormalization(axis=-1)
  classifier.add(LeakyReLU(0.08))
  BatchNormalization(axis=-1)

  classifier.add(MaxPooling2D(pool_size = (2, 2)))
  BatchNormalization(axis=-1)
  classifier.add(Dropout(0.2))
  BatchNormalization(axis=-1)

  classifier.add(Conv2D(128, (3, 3)))
  BatchNormalization(axis=-1)
  classifier.add(LeakyReLU(0.08))
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
  return classifier

#==============================================================================
# creation of two models one with data augmentation and an other without aug..
model_with_aug = final_cnn()
model_without_aug = final_cnn()

#==============================================================================
history_KO_aug = model_without_aug.fit(train_x, train_y,
                                      validation_data= (test_x, test_y),
                                      batch_size = 256, epochs = 25)
history_OK_aug = model_with_aug.fit_generator(training_set,
                         steps_per_epoch = 60000//256,
                         epochs =25,
                         validation_data = test_set,
                         validation_steps = 10000//256)

#==============================================================================
# Saving this final model.
from keras.models import load_model
model_with_aug.save("cnn.h5py")

#==============================================================================
index = ["Without augmentation", "Using augmentation"]
history = [history_KO_aug, history_OK_aug]

#=========================================================================
# visualize overfitting
plot_history(history_KO_aug, index[0])
plot_history(history_OK_aug, index[1])
# plotting average of each size
barplot_acc_loss(index, history)
# plotting the progression of each size during epochs
plot_train_test(index, history, best=1)
# results
show_results(index, history)
#===========================================================================
