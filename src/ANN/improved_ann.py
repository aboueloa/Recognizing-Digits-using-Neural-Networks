#!/usr/bin/env python3
#importing libraries and packages
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

#Evaluating and improving the model
def build_classifier(optimizer, nodes, layer):
  model= tf.keras.Sequential()
  model.add( tf.keras.layers.Dense(layer, kernel_initializer = 'uniform', activation = 'relu', input_dim = 784))
  for i in range(nodes):
    model.add(tf.keras.layers.Dense(layer, kernel_initializer = 'uniform', activation = 'relu'))
  model.add(tf.keras.layers.Dense(10, kernel_initializer = 'uniform', activation = 'softmax'))
  model.compile(optimizer=optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
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
    model = KerasClassifier(build_fn=build_classifier)
    parameters = {'batch_size':[32, 64], 'nb_epoch':[6, 10], 'nodes' : [1, 2, 3], 'layer' : [64, 397], 'optimizer' : ['adam', 'rmsprop']}
    grid_search = GridSearchCV(estimator = model, param_grid = parameters,
                               scoring = 'accuracy', cv = 10)
    grid_search = grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_params_
    best_acc = grid_search.best_score_
    print(best_acc, best_parameters)
main()
