# Keras CNN - OOP Classification Pipeline

The project represents an attempt to build modular, OOP approach
for Multi-class classification on images, which could serve as a template
for other Computer Vision specific tasks. 

This should be viewed as an introductory Computer Vision (CV) project, 
built around Convolutional Neural Network (CNN) architecture in Keras,
 on top of TensorFlow, version 2.3.

The project provides all abstract OOP core features and abstractions 
that could be used in other tasks as well. Besides this, the MNIST 
classification task is implemented as an example of how to use modular
componentes.

## Pipeline Details
The project is separated into the modules which are combined to form
the following pipeline:
1. Dataset loading, batching and prefetching using 'tf.data' Dataset
2. Dataset visualisation: inspection of both original samples from the
    dataset, and the images after the augmentation layer is applied.
3. Model build-up: create the custom architecture specified in the
    configuration file. All parameters (number of ConvLayers,
    existence of Batch Normalization and Pooling layers, etc.) could
    be specified via configuration file.
4. Model training: the whole process is supported by logging tool -
    MLflow, so we are able to track performance across individual
    experiments (where each experiment is denoted with one set of
    the hyper-parameters).
5. Model evaluation on the test dataset: simple accuracy metric.


## Set Up the Project

### Install Necessary Requirements

    make install

### Run Pipeline

    make run

### MLflow Support
In order to track training procedure, the MLflow tracking would log
training parameters, which could be seen on MLflow standalone server.

Run the MLflow UI from the local terminal:

    mlflow ui

## References
* https://github.com/The-AI-Summer/Deep-Learning-In-Production: very 
good materials on Deep Learning in general, including the best 
practices on writing Deep Learning code.