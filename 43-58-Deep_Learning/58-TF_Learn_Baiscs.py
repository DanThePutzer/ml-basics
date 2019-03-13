import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

# Loading and splitting mnist dataset
import tflearn.datasets.mnist as mnist
X, y, testX, testY = mnist.load_data(one_hot=True)

# Reshaping X
X = X.reshape([-1, 28, 28, 1])

# - - Structuring convolutional neural network - -
# Input Layer
convNet = input_data(shape=[None, 28, 28, 1], name='input')
# First Convolutional Layer + Max Pooling
convNet = conv_2d(convNet, 32, 2, activation='relu')
convNet = max_pool_2d(convNet, 2)

# To be continued...