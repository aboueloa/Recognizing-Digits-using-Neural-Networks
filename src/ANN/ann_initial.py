#!/usr/bin/env python3

import numpy as np
import mnist
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

def main():
    #importing data
    train_images = mnist.train_images()
    train_y = mnist.train_labels()
    test_images = mnist.test_images()
    test_y = mnist.test_labels()
    #normalizing data
    train_x = (train_images/255) - 0.5
    test_x = (test_images/255) - 0.5
    train_x = train_x.reshape((-1, 784)) # 28*28 = 784
    test_x = test_x.reshape((-1, 784))
    #initialising the ANN
    model= tf.keras.Sequential()
    #adding the input layer and the first hidden layer
    model.add( tf.keras.layers.Dense(397, activation = 'relu', input_dim = 784))
    #adding the second hidden layer
    model.add(tf.keras.layers.Dense(397, activation = 'relu'))
    #adding the output layer
    model.add(tf.keras.layers.Dense(10, activation = 'softmax'))
    #compiling the ANN
    model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    #Fitting the ANN to the training set
    model.fit(train_x, to_categorical(train_y), epochs = 5, batch_size = 16)
main()
