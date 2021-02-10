# Recognizing Digits using Neural Networks

The goal is to classify an image of a handwritten digit into one of 10 classes
representing integer between 0 and 9.
We use the MNIST dataset of 60000 small 28x28 pixel grayscale images of digits.

## Getting Started

This project assumes that you are using Keras library running on top of Tensorflow
with python 3.


### Installing

The results of all models suggest that GPU is better than CPU when the number
of parameters is high.
So make sure that you install the GPU version of TensorFlow backend:

```
$ pip install tensorflow-gpu
```

### Verification

By default the GPU devices will be given priority if a TensorFlow operation
has both CPU and GPU implementations.
You need to run the ***cpu_gpu.py*** to verify the appearance of both CPU and GPU.

## Building a CNN model

Building models for image classification came with the Convolutional neural network.
Since it involves an enormous number of computations, it is necessary to improve
the CNN by controling the behavior of hyperparameters.

## Optimization

The **src/optimize** folder contains all scripts that we use so as to compare different
models using different classical hyperparameters such as:

* Spatial-extent
* Learning rate
* Loss functions
* Non-linear activation functions
* ...

### Visualization of accuracy and loss over epochs

For example we represent the performance of different types of Pooling using
this Python script:

```
$ cd src/optimize
```
Then :
```
$ ./6_pooling_tuning.py
```
We do the same thing for the other hyperparameter scripts.

### Final CNN model

After choosing the best values for our hyperparameters, we need to save our final
model since this classifier can take hours and even days to train.

Saving models requires that you have the h5py library installed:
```
$ pip install h5py
```
So **src/cnn.h5py** is our last model.

### Evaluating the model

Since we import files from **tools** folder, we have to go to the right **main** folder.
```
$ cd src/main
```
Then we execute the script:
```
$ ./main.py
```
### Running the test

```
$ cd src/test
```
And
```
$ ./test
```

## Authors

* **BENSRHIER Nabil**
* **ABOUELOULA AYMAN**
* **ZRIRA AYOUB**
