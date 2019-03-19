import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import fully_connected, dropout, input_data
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter

# Defining cart pole environment
env = gym.make('CartPole-v0')
env.reset()

learningRate = 0.01
epochCount = 5

# Load training data
  # data[i][0] -> Observations
  # data[i][1] -> Action taken
data = np.load('Data/trainSaved.npy')

# Defining model
def neuralNet(input_size):
  network = input_data(shape=[None, input_size, 1], name='input')

  network = fully_connected(network, 128, activation='relu')
  network = dropout(network, 0.8)

  network = fully_connected(network, 256, activation='relu')
  network = dropout(network, 0.8)

  network = fully_connected(network, 512, activation='relu')
  network = dropout(network, 0.8)

  network = fully_connected(network, 256, activation='relu')
  network = dropout(network, 0.8)

  network = fully_connected(network, 128, activation='relu')
  network = dropout(network, 0.8)

  network = fully_connected(network, 2, activation='softmax')

  network = regression(network, optimizer='adam', learning_rate=learningRate, loss='categorical_crossentropy', name='targets')

  # Defining Model
  model = tflearn.DNN(network) # tensorboard_dir='logs'

  return model

def trainNet(trainingData, model=False):
  # Creating numpy array with all observations
  X = np.array([i[0] for i in trainingData]).reshape(-1, len(trainingData[0][0]), 1)
  # Creating numpy array with all labels
  y = np.array([i[1] for i in trainingData])

  # Checking if model was given
  if not model:
    model = neuralNet(input_size=len(X[0]))
  
  model.fit(
    {'input': X},
    {'targets': y},
    n_epoch=epochCount,
    snapshot_step=500,
    show_metric=True,
    run_id='OpenAiBoi'
  )

  return model

# Training model
finalModel = trainNet(data)

# Saving model
  # Using TFLearn only weights and biases can be saved, not the model's structure
  # To save model's structure use vanilla Tensorflow or Keras
finalModel.save('Model/CartPole.tfl')