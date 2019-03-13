# - General code copied from 46-47-Neural_Network_Model.py -

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

# Defining actual neural network
def recurrentNet(data):
  # Initializing Layers
    # Weights: Determines strength of connections between neurons and, therefore, each neurons influence on the end result
    # Biases: Allows activation function to shift horizontally, giving the model more freedom to fit data
  # A Tensorflow variable represents a tensor of some shape
  hiddenL = {
    'weights': tf.Variable(tf.random_normal([rnnSize, classCount])),
    'biases': tf.Variable(tf.random_normal([classCount]))
  }

  # Modifying x for tensorflow rnn
  data = tf.transpose(data, [1,0,2])
  data = tf.reshape(data, [-1, chunkSize])
  data = tf.split(data, chunkCount, 0)

  lstmCell = rnn.BasicLSTMCell(rnnSize)
  outputs, states = rnn.static_rnn(lstmCell, data, dtype=tf.float32)


  # - Output
  output = tf.matmul(outputs[-1], hiddenL['weights']) + hiddenL['biases']

  return output


# Training neural network
def trainNet(x, y, epochs=10):
  # Initializing model and feeding forward data
  prediction = recurrentNet(x)

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
      epochLoss = 0
      for _ in range(int(mnist.train.num_examples / batchSize)):
        epochX, epochY = mnist.train.next_batch(batchSize)
        epochX = epochX.reshape((batchSize, chunkCount, chunkSize))

        _, c = sess.run([optimizer, cost], feed_dict = {x: epochX, y: epochY})
        epochLoss += c

      print(f'\nEpoch {epoch} out of {epochs}\nLoss: {epochLoss}')

    correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    print('\nAccuracy:', accuracy.eval({x: mnist.test.images.reshape((-1, chunkCount, chunkSize)), y: mnist.test.labels}))


trainNet(x, y, epochs=5)