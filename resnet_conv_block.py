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
        return tf.nn.conv2d(
                X, 
                self.W, 
                strides=[1, self.stride, self.stride, 1], # NHWC format 
                padding=self.padding
                ) + self.b
    
    def copyFromKerasLayers(self, layer):
        W, b = layer.get_weights()
        op1 = self.W.assign(W)
        op2 = self.b.assign(b)
        self.session.run((op1, op2))
        
    def get_params(self):
        return [self.W, self.b]


class BatchNormLayer:
    
    def __init__(self, d):
        self.running_mean = tf.Variable(np.zeros(d, dtype=np.float32), trainable=False) 
        self.running_var = tf.Variable(np.zeros(d, dtype=np.float32), trainable=False)
        self.beta = tf.Variable(np.zeros(d, dtype=np.float32)) # offset
        self.gamma = tf.Variable(np.zeros(d, dtype=np.float32)) # scale

    def forward(self, X):
        return tf.nn.batch_normalization(
            X,
            self.running_mean,
            self.running_var,
            self.beta,
            self.gamma,
            1e-3 # variance_epsilon
            )
    
    def copyFromKerasLayers(self, layer):
        gamma, beta, running_mean, running_var = layer.get_weights()
        op1 = self.running_mean.assign(running_mean)
        op2 = self.running_var.assign(running_var)
        op3 = self.gamma.assign(gamma)
        op4 = self.beta.assign(beta)
        self.session.run((op1, op2, op3, op4))
                    
    def get_params(self):
        return [self.running_mean, self.running_var, self.beta, self.gamma]
                
    
class ConvBlock:

    def __init__(self, in_channels=64, layer_sizes=[64, 64, 256], stride=2, activation=tf.nn.relu):
        self.session = None
        self.activation = activation
        self.input_ = tf.placeholder(tf.float32, shape=(1, 56, 56, in_channels))

        # init main branch
        self.conv1 = ConvLayer(1, 1, in_channels, layer_sizes[0], stride)
        self.bn1   = BatchNormLayer(layer_sizes[0])
        self.conv2 = ConvLayer(3, 3, layer_sizes[0], layer_sizes[1], 1, 'SAME')
        self.bn2   = BatchNormLayer(layer_sizes[0])
        self.conv3 = ConvLayer(1, 1, layer_sizes[1], layer_sizes[2], 1)
        self.bn3   = BatchNormLayer(layer_sizes[2])

        # init shortcut branch
        self.convs = ConvLayer(1, 1, in_channels, layer_sizes[2], stride)
        self.bns   = BatchNormLayer(layer_sizes[2])
        
        # for later use
        self.layers = [
                self.conv1, self.bn1,
                self.conv2, self.bn2,
                self.conv3, self.bn3,
                self.convs, self.bns
                ]
        
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
        
        # shortcut branch
        shortcut_X = self.convs.forward(X)
        shortcut_X = self.bns.forward(shortcut_X)
        
        return self.activation(main_X + shortcut_X)

    def predict(self, X):
        if self.session:
            return self.session.run(
                self.forward(self.input_),
                feed_dict={self.input_: X})
        else:
            print("Session is not active!")
    
    def set_session(self, session):
    # so that assignments happen on sublayers
        self.session = session
        self.conv1.session = session
        self.bn1.session = session
        self.conv2.session = session
        self.bn2.session = session
        self.conv3.session = session
        self.bn3.session = session
        self.convs.session = session
        self.bns.session = session
    
    def copyFromKerasLayers(self, layers):
        self.conv1.copyFromKerasLayers(layers[0])
        self.bn1.copyFromKerasLayers(layers[1])
        self.conv2.copyFromKerasLayers(layers[3])
        self.bn2.copyFromKerasLayers(layers[4])
        self.conv3.copyFromKerasLayers(layers[6])
        self.bn3.copyFromKerasLayers(layers[8])
        self.convs.copyFromKerasLayers(layers[7])
        self.bns.copyFromKerasLayers(layers[9])

    def get_params(self):
        params = []
        for layer in self.layers:
            params += layer.get_params()
        return params


if __name__ == '__main__':
    conv_block = ConvBlock()

    # make a fake image
    X = np.random.random((1, 56, 56, 64))

    init = tf.global_variables_initializer()
    with tf.Session() as session:
        conv_block.set_session(session)
        session.run(init)

        output = conv_block.predict(X)
        print("output.shape:", output.shape)
