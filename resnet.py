import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras

from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.preprocessing import image
from keras.layers import Dense
from keras.applications.resnet50 import preprocess_input, decode_predictions

from resnet_conv_block import ConvLayer, BatchNormLayer, ConvBlock
from resnet_identity_block import IdentityBlock
from utils import ReLULayer, MaxPoolLayer, AvgPool, Flatten, custom_softmax


class DenseLayer:
    
    def __init__(self, in_channels, out_channels):
        self.W = tf.Variable((np.random.randn(in_channels, out_channels) * \
                              np.sqrt(2.0 / in_channels)).astype(np.float32))
        self.b = tf.Variable(np.zeros(out_channels, dtype=np.float32))
        
    def forward(self, X):
        return tf.matmul(X, self.W) + self.b
    
    def copyFromKerasLayers(self, layer):
        W, b = layer.get_weights()
        op1 = self.W.assign(W)
        op2 = self.b.assign(b)
        self.session.run((op1, op2))
        
    def get_params(self):
        return [self.W, self.b]
    
    
class ResNet:
    
    def __init__(self):
        self.layers = [
            # before conv block
            ConvLayer(filter_height=7, filter_width=7, in_channels=3, out_channels=64, stride=2, padding='SAME'),
            BatchNormLayer(64),
            ReLULayer(),
            MaxPoolLayer(length=3),
            # conv block
            ConvBlock(in_channels=64, layer_sizes=[64, 64, 256], stride=1),
            # identity block x 2
            IdentityBlock(in_channels=256, layer_sizes=[64, 64, 256]),
            IdentityBlock(in_channels=256, layer_sizes=[64, 64, 256]),
            # conv block
            ConvBlock(in_channels=256, layer_sizes=[128, 128, 512], stride=2),
            # identity block x 3
            IdentityBlock(in_channels=512, layer_sizes=[128, 128, 512]),
            IdentityBlock(in_channels=512, layer_sizes=[128, 128, 512]),
            IdentityBlock(in_channels=512, layer_sizes=[128, 128, 512]),
            # conv block
            ConvBlock(in_channels=512, layer_sizes=[256, 256, 1024], stride=2),
            # identity block x 5
            IdentityBlock(in_channels=1024, layer_sizes=[256, 256, 1024]),
            IdentityBlock(in_channels=1024, layer_sizes=[256, 256, 1024]),
            IdentityBlock(in_channels=1024, layer_sizes=[256, 256, 1024]),
            IdentityBlock(in_channels=1024, layer_sizes=[256, 256, 1024]),
            IdentityBlock(in_channels=1024, layer_sizes=[256, 256, 1024]),
            # conv block
            ConvBlock(in_channels=1024, layer_sizes=[512, 512, 2048], stride=2),
            # identity block x 2
            IdentityBlock(in_channels=2048, layer_sizes=[512, 512, 2048]),
            IdentityBlock(in_channels=2048, layer_sizes=[512, 512, 2048]),
            # pool / flatten / dense
            AvgPool(length=7),
            Flatten(),
            DenseLayer(in_channels=2048, out_channels=1000)
            ]
        
        self.input_ = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
        self.output = self.forward(self.input_)
        
    def copyFromKerasLayers(self, layers):
        # conv layer
        self.layers[0].copyFromKerasLayers(layers[1])
        # batch norm layer
        self.layers[1].copyFromKerasLayers(layers[2])
        # conv block
        self.layers[4].copyFromKerasLayers(layers[5:17])
        # identity block x 2
        self.layers[5].copyFromKerasLayers(layers[17:27])
        self.layers[6].copyFromKerasLayers(layers[27:37])
        # conv block
        self.layers[7].copyFromKerasLayers(layers[37:49])
        # identity block x 3
        self.layers[8].copyFromKerasLayers(layers[49:59])
        self.layers[9].copyFromKerasLayers(layers[59:69])
        self.layers[10].copyFromKerasLayers(layers[69:79])
        # conv block
        self.layers[11].copyFromKerasLayers(layers[79:91])
        # identity block x 5
        self.layers[12].copyFromKerasLayers(layers[91:101])
        self.layers[13].copyFromKerasLayers(layers[101:111])
        self.layers[14].copyFromKerasLayers(layers[111:121])
        self.layers[15].copyFromKerasLayers(layers[121:131])
        self.layers[16].copyFromKerasLayers(layers[131:141])
        # conv bloc
        self.layers[17].copyFromKerasLayers(layers[141:153])
        # identity block x 2
        self.layers[18].copyFromKerasLayers(layers[153:163])
        self.layers[19].copyFromKerasLayers(layers[163:173])
        # dense
        self.layers[22].copyFromKerasLayers(layers[175])
        
    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    def predict(self, X):
        if self.session:
            return self.session.run(
                    self.output,
                    feed_dict={self.input_: X}
                    )
        print("Session is not active!")
        
    def set_session(self, session):
        self.session = session
        for layer in self.layers:
            if isinstance(layer, ConvBlock) or isinstance(layer, IdentityBlock):
                layer.set_session(session)
            else:
                layer.session = session
    
    def get_params(self):
        params = []
        for layer in self.layers:
            params += layer.get_params()
        

if __name__ == '__main__':
    resnet = ResNet50(weights='imagenet')
    
    # a new resnet without softmax
    x = resnet.layers[-2].output
    W, b = resnet.layers[-1].get_weights()
    y = Dense(1000)(x)
    resnet = Model(resnet.input, y)
    resnet.layers[-1].set_weights([W, b])
    
    # take a part of the model
    partial_model = Model(
            inputs=resnet.input,
            outputs=resnet.layers[175].output
            )
    
    print(partial_model.summary())
    
    # create an instance of our own model
    my_partial_resnet = ResNet()
    
    
    # make a fake image
    X = np.random.random((1, 224, 224, 3))
    
    # get Keras output
    keras_output = partial_model.predict(X)
    
    # define varible initializer
    init = tf.variables_initializer(my_partial_resnet.get_params())
    
    # get keras session and run initialization
    session = keras.backend.get_session()
    my_partial_resnet.set_session(session)
    session.run(init)
    
    # copy params from Keras model
    my_partial_resnet.copyFromKerasLayers(partial_model.layers)
    
    # get output of my model
    my_output = my_partial_resnet.predict(X)
    
    # compare 2 models
    diff = np.abs(my_output - keras_output).sum()
    if diff < 1e-10:
        print("Everything's great!")
    else:
        print("diff = {:.3f}".format(diff))
