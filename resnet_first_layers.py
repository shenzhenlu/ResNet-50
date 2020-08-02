# Let's go up to the end of the first conv block
# to make sure everything has been loaded correctly
# compared to keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras

from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions

from resnet_convblock import ConvLayer, BatchNormLayer, ConvBlock

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


class PartialResNet:
    def __init__(self):
        self.input_ = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
        
        self.conv = ConvLayer(7, 7, 3, 64, 2, 'SAME')
        self.bn = BatchNormLayer(64)
        self.relu = ReLULayer()
        self.mp = MaxPoolLayer(dim=3)
      
        self.cb = ConvBlock(in_channels=64, layer_sizes=[64, 64, 256], stride=1)
        
    def forward(self, X):
        X = self.conv.forward(X)
        X = self.bn.forward(X)
        X = self.relu.forward(X)
        X = self.mp.forward(X)
        
        return self.cb.forward(X)
    
    def predict(self, X):
        if self.session:
            return self.session.run(
                self.forward(self.input_),
                feed_dict={self.input_: X})
        print("Session is not active")

    def set_session(self, session):
        self.session = session
    
    def get_params(self):
        param_list = [self.conv.get_params(), 
                  self.bn.get_params(), 
                  self.relu.get_params(), 
                  self.mp.get_params(),
                  self.cb.get_params()]
        
        return param_list
          
        
if __name__ == '__main__':
    resnet = ResNet50(weights='imagenet')
    
    # you can determine the correct layer
    # by looking at resnet.layers in the console
    partial_model = Model(
      inputs=resnet.input,
      outputs=resnet.layers[16].output
    )
    print(partial_model.summary())
    # for layer in partial_model.layers:
    #   layer.trainable = False
    
    my_partial_resnet = PartialResNet()
    
    # make a fake image
    X = np.random.random((1, 224, 224, 3))
    
    # get keras output
    keras_output = partial_model.predict(X)
    
    # get my model output
    init = tf.variables_initializer(my_partial_resnet.get_params())
    
    # note: starting a new session messes up the Keras model
    session = keras.backend.get_session()
    my_partial_resnet.set_session(session)
    session.run(init)
    
    # first, just make sure we can get any output
    first_output = my_partial_resnet.predict(X)
    print("first_output.shape:", first_output.shape)
    
    # copy params from Keras model
    my_partial_resnet.copyFromKerasLayers(partial_model.layers)
    
    # compare the 2 models
    output = my_partial_resnet.predict(X)
    diff = np.abs(output - keras_output).sum()
    if diff < 1e-10:
      print("Everything's great!")
    else:
      print("diff = %s" % diff)
