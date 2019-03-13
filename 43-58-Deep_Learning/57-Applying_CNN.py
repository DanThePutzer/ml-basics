# - General code copied from 55-Applying_RNN.py -

import tensorflow as tf
# Getting mnist dataset
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn

mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

# Defining general parameters
classCount = 10
batchSize = 128

chunkSize = 28
chunkCount = 28
rnnSize = 128

# Creating placeholder matrices for layers
# If placeholder is given a shape it can help prevent feeding wrong shape of data because it will throw and error if data is not in specified shape
# x -> Input layer: Basically just an array with each pixel's data (Images are 28x28 which equals 784 values)
# [None, nodeCount_Input]
x = tf.placeholder('float', [None, chunkCount, chunkSize])
y = tf.placeholder('float')

def conv2D(x, w):
  return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')

def maxpool2D(x):
  # ksize -> Size of pooling window
  # strides -> Size of steps
  return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# Defining actual neural network
def convolutionalNet(data):
  # Initializing Layers
    # Weights: Determines strength of connections between neurons and, therefore, each neurons influence on the end result
    # Biases: Allows activation function to shift horizontally, giving the model more freedom to fit data
  # A Tensorflow variable represents a tensor of some shape
  weights = {
    # 5x5 convolution window, takes 1 input channel, produces 32 outputs channels (features)
    # Window size and output channels are set by user and have to be experimented with
    # Input channels depend on image, only 1 channel here since mnist images are in greyscale (would use 3 for RGB images)
    'wConv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    'wConv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # Images are 28x28, goes through 2 pooling steps which cut resolution in half: 28 -> 14 -> 7 therefore 7*7
    'wFullCon': tf.Variable(tf.random_normal([7*7*64, 1024])),
    'wOutput': tf.Variable(tf.random_normal([1024, classCount])),
  }

  biases = {
    'bConv1': tf.Variable(tf.random_normal([32])),
    'bConv2': tf.Variable(tf.random_normal([64])),
    'bFullCon': tf.Variable(tf.random_normal([1024])),
    'bOutput': tf.Variable(tf.random_normal([classCount])),
  }

  # Turning 784x1 image into a 28x28 image
  data = tf.reshape(data, [-1, 28, 28, 1])

  conv1 = tf.nn.relu(conv2D(data, weights['wConv1']) + biases['bConv1'])
  conv1 = maxpool2D(conv1)

  conv2 = tf.nn.relu(conv2D(conv1, weights['wConv2']) + biases['bConv2'])
  conv2 = maxpool2D(conv2)

  fullCon = tf.reshape(conv2, [-1, 7*7*64])
  fullCon = tf.nn.relu(tf.matmul(fullCon, weights['wFullCon']) + biases['bFullCon'])

  # Dropout ignores some randomly selected neurons on each iteration
    # Neurons can develop co-dependency on each other which can lead to overfitting, dropout should help with that
    # Each iteration is performed with a different, random network structure
  fullCon = tf.nn.dropout(fullCon, rate=0.2)

  output = tf.matmul(fullCon, weights['wOutput']) + biases['bOutput']

  return output


# Training neural network
def trainNet(x, y, epochs=10):
  # Initializing model and feeding forward data
  prediction = convolutionalNet(x)

  # Computing cost
    # tf.reduce_mean: Reduces dimensionality of tensor along axis by taking mean of values
    # Softmax function would usually normalize output values (-> returns values between 0 and 1, could be seen as probablilities)
    # -> Softmax is a more general version of Sigmoid
    # with_logits tells it that it should not normalize (or something like that...)
    # Cross entropy basically calculates the distance between the output and the label
  cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y) )

  # Optimizing cost function
  optimizer = tf.train.AdamOptimizer().minimize(cost)

  # Starting Tensorflow session
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
      print(f'\nEpoch {epoch} out of {epochs}')
      epochLoss = 0
      for _ in range(int(mnist.train.num_examples / batchSize)):
        epochX, epochY = mnist.train.next_batch(batchSize)
        epochX = epochX.reshape((batchSize, chunkCount, chunkSize))

        _, c = sess.run([optimizer, cost], feed_dict = {x: epochX, y: epochY})
        epochLoss += c

      print(f'Loss: {epochLoss}\n')

    correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    print('\nAccuracy:', accuracy.eval({x: mnist.test.images.reshape((-1, chunkCount, chunkSize)), y: mnist.test.labels}))


trainNet(x, y, epochs=9)