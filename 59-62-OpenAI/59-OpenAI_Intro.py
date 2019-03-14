import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import fully_connected, dropout, input_data
from tflearn.layers.conv import conv_2d, max_pool_2d
from statistics import mean, median
from collections import Counter

# Defining environment with cart pole game
env = gym.make('CartPole-v0')
env.reset()

# Defining basic parameters
learningRate = 1e-3
goalSteps = 500
scoreReq = 50
initialGames = 10000

# Messing around with random actions to test Gym
def randomAction():
  # episode = epoch
  for episode in range(100):
    env.reset()
    for i in range(goalSteps):
      env.render()
      # Creating random action to take
      action = env.action_space.sample()
      # Performing random action for each step
      observation, reward, done, info = env.step(action)
      if done:
        break

randomAction()