import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class ReLULayer:
    
    def forward(self, X):
        return tf.nn.relu(X)
    
    def get_params(self):
        return []
    
    
class MaxPoolLayer:
    
    def __init__(self, length):
        self.length = length
        
    def forward(self, X):
        return tf.nn.max_pool(
            X,
            ksize=[1, self.length, self.length, 1],
            strides=[1, 2, 2, 1],
            padding='VALID')
        
    def get_params(self):
        return []

class AvgPool:
    
    def __init__(self, length):
        self.length = length
        
    def forward(self, X):
        return tf.nn.avg_pool(
                X,
                ksize=[1, self.length, self.length, 1],
                strides=[1, 1, 1, 1],
                padding='VALID'
                )
        
    def get_params(self):
        return []
    
    
class Flatten:
    
    def forward(self, X):
        return tf.contrib.layers.flatten(X)
    
    def get_params(self):
        return []
    
    
def custom_softmax(x):
    m = tf.reduce_max(x, 1)
    x = x - m
    e = tf.exp(x)
    
    return e / tf.reduce_sum(e, 1)


