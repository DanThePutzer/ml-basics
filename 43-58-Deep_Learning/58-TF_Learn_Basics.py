import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

# Loading and splitting mnist dataset
import tflearn.datasets.mnist as mnist
X, Y, testX, testY = mnist.load_data(one_hot=True)

# Reshaping X
X = X.reshape([-1, 28, 28, 1])
testX = testX.reshape([-1, 28, 28, 1])

# - - Structuring convolutional neural network - -
# Input Layer
convNet = input_data(shape=[None, 28, 28, 1], name='input')
# First Convolutional Layer + Max Pooling
  # Tensor to perform convolution on, number of filters (aka. number of features put out by layer), filter size
convNet = conv_2d(convNet, 32, 2, activation='relu')
convNet = max_pool_2d(convNet, 2)

# Second Convolutional Layer + Max Pooling
convNet = conv_2d(convNet, 64, 2, activation='relu')
convNet = max_pool_2d(convNet, 2)

# Fully connected layer
  # 1024 = number of neurons in layer
convNet = fully_connected(convNet, 1024, activation='relu')

# Dropout layer
  # 0.8 = Keep rate
convNet = dropout(convNet, 0.8)

# Output layer
convNet = fully_connected(convNet, 10, activation='softmax')
convNet = regression(convNet, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy', name='targets')

# Defining model
model = tflearn.DNN(convNet)

# Fitting model
model.fit(
  {'input': X},{'targets': Y},
  n_epoch=3,
  validation_set=({'input': testX},{'targets': testY}),
  snapshot_step=500,
  show_metric=True,
  run_id='mnistModel'
)

# Saving model weights only (not whole model like pickle would)
# model.save('TFlearn-Mnist.model')