# - Quick Summary of Architecture:
# Input Layer > Get Weighted > Hidden Layer 1 > passed thru (Activiation Function)
# Results from Layer 1 > Weights > Hidden Layer 2 > passed thru (Activation Function)
# Results from Layer 2 > Weights > Output Layer
# Called feed forward -> Passing information forward through layers

# - Evaluation:
# Use Cost Function to evaluate how good model output compares to intended output
# -> Example cost function: Cross Entropy
# Minimize cost with Optimization Function
# -> Example optimization function: AdamOptimizer, AdaGrad 

# - Improvement:
# Use backpropagation to update weights based on learned data

# Each feed forward and backpropagation form one epoch

import tensorflow as tf
# Getting mnist dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)
print('\n- - - -\n')
print(mnist)
print('\n- - - -\n')

# Defining node count for each of the layers
nodeCount_Input = 784
nodeCount_l1 = 500
nodeCount_l2 = 500
nodeCount_l3 = 500

# Defining number of desired output classes
classCount = 10
# Defining batchsize -> Determines how many datapoints (in this case images) are fed through neural net at a time
  # -> Becomes important when dataset is to large to be loaded into memory all at once
batchSize = 100

# Creating placeholder matrices for layers
# If placeholder is given a shape it can help prevent feeding wrong shape of data because it will throw and error if data is not in specified shape
# x -> Input layer: Basically just an array with each pixel's data (Images are 28x28 which equals 784 values)
x = tf.placeholder('float', [None, nodeCount_Input])
y = tf.placeholder('float')

# Defining actual neural network
def modelNet(data):
  # Initializing Layers
    # Weights: Determines strength of connections between neurons and, therefore, each neurons influence on the end result
    # Biases: Allows activation function to shift horizontally, giving the model more freedom to fit data
  # A Tensorflow variable represents a tensor of some shape
  hiddenL1 = {
    'weights': tf.Variable(tf.random_normal([nodeCount_Input, nodeCount_l1])),
    'biases': tf.Variable(tf.random_normal([nodeCount_l1]))
  }

  hiddenL2 = {
    'weights': tf.Variable(tf.random_normal([nodeCount_l1, nodeCount_l2])),
    'biases': tf.Variable(tf.random_normal([nodeCount_l2]))
  }

  hiddenL3 = {
    'weights': tf.Variable(tf.random_normal([nodeCount_l2, nodeCount_l3])),
    'biases': tf.Variable(tf.random_normal([nodeCount_l3]))
  }

  outputL = {
    'weights': tf.Variable(tf.random_normal([nodeCount_l3, classCount])),
    'biases': tf.Variable(tf.random_normal([classCount]))
  }

  # Defining actual model:
  # - Layer 1
  # layer = sum[ data from previous layer * weights + biases ]
  l1 = tf.add(tf.matmul(data, hiddenL1['weights']), hiddenL1['biases'])
  # Passing through Activation function 
    # -> relu = Rectified Linear, gives 0 for all negative values and gives input as output for all positive values
  l1 = tf.nn.relu(l1)

  # - Layer 2
  l2 = tf.add(tf.matmul(l1, hiddenL2['weights']), hiddenL2['biases'])
  l2 = tf.nn.relu(l2)

  # - Layer 3
  l3 = tf.add(tf.matmul(l2, hiddenL3['weights']), hiddenL3['biases'])
  l3 = tf.nn.relu(l3)

  # - Output
  output = tf.matmul(l3, outputL['weights']) + outputL['biases']

  return output


# Training neural network
def trainNet(x, y, epochs=10):
  # Initializing model and feeding forward data
  prediction = modelNet(x)

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
        _, c = sess.run([optimizer, cost], feed_dict = {x: epochX, y: epochY})
        epochLoss += c

      print(f'\nEpoch {epoch} out of {epochs}\nLoss: {epochLoss}')

    correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    print('\nAccuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


trainNet(x, y, epochs=15)