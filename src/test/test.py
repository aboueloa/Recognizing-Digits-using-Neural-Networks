#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# load the model
from keras.models import load_model
classifier = load_model("../cnn.h5py")
# import the input image
from keras.preprocessing import image
import cv2
# On a 3 images on choisit une.
def predict(path_image):
    ii = cv2.imread(path_image)
    ii = cv2.resize(ii, (28, 28))
    # convert the image to gray one
    test_image = cv2.cvtColor(ii, cv2.COLOR_BGR2GRAY)
    test_image = image.img_to_array(test_image)
    # normalizing
    test_image = test_image.astype('float32')
    test_image = (255 - test_image)/255
    # reshape to (28, 28, 1)
    test_image = np.expand_dims(test_image, axis = 0)
    # predict the digit
    result = classifier.predict(test_image)
    # plot the image with the predicted digit
    plt.imshow(test_image.reshape(28,28))
    plt.title(" The predicted digit is : " +
              str(np.argmax(result,axis = 1)[0]),fontweight="bold")
    plt.show()

predict("4.png")
predict("7.png")
predict("9.png")
