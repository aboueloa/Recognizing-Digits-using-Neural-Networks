#!/usr/bin/env python3
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import mnist
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
def roc(y_test, y_score):
    lw = 2
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = 10
    for i in range(10):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()


def plot_pr(y, y_hat):
    plt.style.use('seaborn-darkgrid')
    lw = 2
    n_classes = 10
    fpr, tpr, roc_auc = dict(), dict(), dict()

    for i in range(10):
       fpr[i], tpr[i], _ = precision_recall_curve(y[:, i], y_hat[:, i])

    for i in range(n_classes):
       plt.plot(fpr[i], tpr[i], lw=lw,
               label='Digit {}'.format(i))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontweight="bold")
    plt.ylabel('Precision', fontweight="bold")
    plt.title('Precision-Recall ', fontweight="bold")
    plt.legend(loc="lower right", prop={"weight":"bold"})
    plt.show()

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
    #model
    model= tf.keras.Sequential()
    model.add( tf.keras.layers.Dense(397, activation = 'relu', input_dim = 784))
    model.add(tf.keras.layers.Dense(397, activation = 'relu'))
    model.add(tf.keras.layers.Dense(10, activation = 'softmax'))
    model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    model.fit(train_x, to_categorical(train_y), epochs = 10, batch_size = 32)
    y_test = tf.keras.utils.to_categorical(tf.keras.datasets.mnist.load_data()[1][1], num_classes=10, dtype='float32')
    y_score = model.predict_proba(test_x)
    plot_pr(y_test, y_score)
    roc(y_test, y_score)
