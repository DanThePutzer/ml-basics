import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import fully_connected, dropout, input_data
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

# Initializing things
env = gym.make('CartPole-v0')
env.reset()
learningRate = 0.01
goalSteps = 500

# - - - Defining model - - - 
  # (Structure copied from 61-Training_Model_OpenAI.py since TFLearn cannot save a model's structure)
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
  model = tflearn.DNN(network)

  # Loading model parameters from training
  model.load('Model/CartPole.tfl')

  return model


scores = []
predictions = []

model = neuralNet(input_size=4)

# Playing a few rounds
for game in range(10):
  score = 0
  gameMemory = []
  prevObs = []

  env.reset()

  for _ in range(goalSteps):
    env.render()
    if len(prevObs) == 0:
      action = env.action_space.sample()
    else:
      action = np.argmax(model.predict(prevObs.reshape(-1, len(prevObs),1))[0])

    newObs, reward, done, info = env.step(action)
    prevObs = newObs
    score += reward

    gameMemory.append([newObs, action])

    if done:
      break
  
  scores.append(score)

print(f'Average Score: {sum(scores)/len(scores)}')

