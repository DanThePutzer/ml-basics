import tensorflow as tf
import numpy as np
import pickle

# Load data from pickle
trainX, trainY, testX, testY = pickle.load( open('Data/SentimentSet.pickle', 'rb'))

# Defining parameters for neural net
nodeCount_Input = len(trainX[0])
nodeCount_l1 = 500
nodeCount_l2 = 500
nodeCount_l3 = 500

classCount = 2
batchSize = 100

# Creating placeholder matrices for layers
x = tf.placeholder('float')
y = tf.placeholder('float')

# Initializing layers
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

# Defining neural net
def modelNet(data):
  
  l1 = tf.add(tf.matmul(data, hiddenL1['weights']), hiddenL1['biases'])
  l1 = tf.nn.relu(l1)

  l2 = tf.add(tf.matmul(l1, hiddenL2['weights']), hiddenL2['biases'])
  l2 = tf.nn.relu(l2)

  l3 = tf.add(tf.matmul(l2, hiddenL3['weights']), hiddenL3['biases'])
  l3 = tf.nn.relu(l3)

  output = tf.matmul(l3, outputL['weights']) + outputL['biases']

  return output

# Training neural net
def trainNet(x, epochs=10):
  prediction = modelNet(x)

  cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y) )
  optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
      epochLoss = 0

      i = 0
      while i < len(trainX):
        start = i
        stop = i + batchSize

        batchX = np.array(trainX[start:stop])
        batchY = np.array(trainX[start:stop])

        _, c = sess.run([optimizer, cost], feed_dict={x: batchX, y: batchY})
        epochLoss += c
        i += batchSize

      print(f'\nEpoch {epoch} out of {epochs}\nLoss: {epochLoss}')

    correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    print('\nAccuracy:', accuracy.eval({x: testX, y: testY}))

trainNet(x)
