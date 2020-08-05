#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Go up to the end of the first conv block to make 
sure everything has been loaded correctly
"""


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras

from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions

from resnet_conv_block import ConvLayer, BatchNormLayer, ConvBlock


class ReLULayer:
    
    def forward(self, X):
        return tf.nn.relu(X)
    
    def get_params(self):
        return []
    
    
class MaxPoolLayer:
    
    def __init__(self, dim):
        self.dim = dim
        
    def forward(self, X):
        return tf.nn.max_pool(
            X,
            ksize=[1, self.dim, self.dim, 1],
            strides=[1, 2, 2, 1],
            padding='VALID')
        
    def get_params(self):
        return []


class FirstLayers:
    
    def __init__(self):
        self.input_ = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
        self.session = None
        self.layers = [        
            ConvLayer(7, 7, 3, 64, 2, 'SAME'),
            BatchNormLayer(64),
            ReLULayer(),
            MaxPoolLayer(dim=3),
            ConvBlock(in_channels=64, layer_sizes=[64, 64, 256], stride=1)
            ]
    
    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        
        return X
    
    def predict(self, X):
        if self.session:
            return self.session.run(
                self.forward(self.input_),
                feed_dict={self.input_: X}
                )
            
        print("Session is not active")

    def set_session(self, session):
        self.session = session
        self.layers[0].session = session
        self.layers[1].session = session
        self.layers[4].set_session(session)
        
    def copyFromKerasLayers(self, layers):
        self.layers[0].copyFromKerasLayers(layers[1])
        self.layers[1].copyFromKerasLayers(layers[2])
        self.layers[4].copyFromKerasLayers(layers[5:])
    
    def get_params(self):
        params = []
        for layer in self.layers:
            params += layer.get_params()
            
        return params
        
    
if __name__ == '__main__':
    resnet = ResNet50(weights='imagenet')
    
    keras_first_layers = Model(
            inputs=resnet.input,
            outputs=resnet.layers[16].output
            )
    
    print(keras_first_layers.summary())
    
    # make a fake image
    X = np.random.random((1, 224, 224, 3))

    ## get keras output
    keras_output = keras_first_layers.predict(X)    
    
    # get my model output
    my_first_layers = FirstLayers()
    init = tf.variables_initializer(my_first_layers.get_params())
    
    # get the current Tensorflow session
    session = keras.backend.get_session()
    my_first_layers.set_session(session)
    session.run(init)

    # make sure we can get any output
    first_output = my_first_layers.predict(X)

    print("first_output.shape:", first_output.shape)

    # copy params from Keras model
    my_first_layers.copyFromKerasLayers(keras_first_layers.layers)

    # get our output
    my_output = my_first_layers.predict(X)
    
    # close session
    session.close()

    # compare the 2 models
    diff = np.abs(my_output - keras_output).sum()
    if diff < 1e-10:
        print("Everything's great!")
    else:
        print("diff = {:.3f}".format(diff))
        