import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tf_resnet_convblock import ConvLayer, BatchNormLayer


class IdentityBlock:
    """
    """
    def __init__(self, in_channels=3, layer_sizes=[64, 64, 256], activation=tf.nn.relu):
        self.session = None
        self.activation = activation
        self.input_ = tf.placeholder(tf.float32, shape=(1, 224, 224, in_channels))

        # init main branch
        self.conv1 = ConvLayer(1, 1, in_channels, layer_sizes[0], 1)
        self.bn1   = BatchNormLayer(layer_sizes[0])
        self.conv2 = ConvLayer(3, 3, layer_sizes[0], layer_sizes[1], 1, 'SAME')
        self.bn2   = BatchNormLayer(layer_sizes[0])
        self.conv3 = ConvLayer(1, 1, layer_sizes[1], layer_sizes[2], 1)
        self.bn3  = BatchNormLayer(layer_sizes[2])

    def forward(self, X):
        # main branch
        main_X = self.conv1.forward(X)
        main_X = self.bn1.forward(main_X)
        main_X = self.activation(main_X)
        
        main_X = self.conv2.forward(main_X)
        main_X = self.bn2.forward(main_X)
        main_X = self.activation(main_X)
        
        main_X = self.conv3.forward(main_X)
        main_X = self.bn3.forward(main_X)
        
        return self.activation(main_X + X)

    def predict(self, X):
        if self.session:
            return self.session.run(
                self.forward(self.input_),
                feed_dict={self.input_: X})
        else:
            print("Session is not active!")


if __name__ == '__main__':
  identity_block = IdentityBlock()

  # make a fake image
  X = np.random.random((1, 224, 224, 256))

  init = tf.global_variables_initializer()
  with tf.Session() as session:
    #identity_block.set_session(session)
    identity_block.session = session
    session.run(init)

    output = identity_block.predict(X)
    print("output.shape:", output.shape)
