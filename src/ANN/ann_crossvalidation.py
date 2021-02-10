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

def build_classifier():
  model= tf.keras.Sequential()
  model.add( tf.keras.layers.Dense(397, kernel_initializer = 'uniform', activation = 'relu', input_dim = 784))
  model.add(tf.keras.layers.Dense(397, kernel_initializer = 'uniform', activation = 'relu'))
  model.add(tf.keras.layers.Dense(10, kernel_initializer = 'uniform', activation = 'softmax'))
  model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
  return model

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
    classifier = KerasClassifier(build_fn=build_classifier, batch_size = 16, nb_epoch = 5)
    accuracies = cross_val_score(estimator = classifier, X = train_x, y = train_y, cv = 10)
    mean = accuracies.mean()
    print(mean)
    print(accuracies.std())
main()
