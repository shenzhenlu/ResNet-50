import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def he_normal_init(filter_height, filter_width, in_channels, out_channels, stride):
    return (np.random.randn(filter_height, filter_width, in_channels, out_channels)
    * np.sqrt(2.0 / (filter_height * filter_width * in_channels))).astype(np.float32)


class ConvLayer:
    def __init__(self, filter_height, filter_width, in_channels, out_channels, stride, padding='VALID'):
        self.W = tf.Variable(he_normal_init(filter_height, filter_width, in_channels, out_channels, stride))
        self.b = tf.Variable(np.zeros(out_channels, dtype=np.float32))
        self.stride = stride
        self.padding = padding
    
    def forward(self, X):
        return tf.nn.conv2d(X,
                        self.W,
                        strides=[1, self.stride, self.stride, 1], # NHWC format
                        padding=self.padding
                        ) + self.b
    
    def get_params(self):
        return [self.W, self.b]


class BatchNormLayer:
    def __init__(self, d):
        self.running_mean = tf.Variable(np.zeros(d, dtype=np.float32), trainable=False) 
        self.running_var = tf.Variable(np.zeros(d, dtype=np.float32), trainable=False)
        self.beta = tf.Variable(np.zeros(d, dtype=np.float32), trainable=False) # offset
        self.gamma = tf.Variable(np.zeros(d, dtype=np.float32), trainable=False) # scale

    def forward(self, X):
        return tf.nn.batch_normalization(
            X,
            self.running_mean,
            self.running_var,
            self.beta,
            self.gamma,
            1e-3 # variance_epsilon
            )
    
    def get_params(self):
        return [self.running_mean, self.running_var, self.beta, self.gamma]


class ConvBlock:
    """
    """
    def __init__(self, in_channels, conv_layers, stride=2, activatiob=tf.nn.relu):
        self.session = None
        self.f = tf.nn.relu

        # init main branch
        self.conv1 = ConvLayer(1, 1, in_channels, con_layers[0], stride)
        self.bn1   = BatchNormLayer(conv_layers[0])
        self.conv2 = ConvLayer(3, 3, con_layers[0], con_layers[1], 1, 'SAME')
        self.bn1   = BatchNormLayer(conv_layers[0])
        self.conv1 = ConvLayer(1, 1, con_layers[1], con_layers[2], 1)
        self.bn1   = BatchNormLayer(conv_layers[0])

    def predict(self, X):
      pass


if __name__ == '__main__':
  conv_block = ConvBlock()


  # make a fake image
  X = np.random.random((1, 224, 224, 3))

  init = tf.global_variables_initializer()
  with tf.Session() as session:
    conv_block.session = session
    session.run(init)

    output = conv_block.predict(X):
    print("output.shape:", output.shape)