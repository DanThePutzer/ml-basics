import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import fully_connected, dropout, input_data
from tflearn.layers.conv import conv_2d, max_pool_2d
from statistics import mean, median
from collections import Counter

# Defining cart pole environment
env = gym.make('CartPole-v0')
env.reset()

# Defining parameters
goalSteps = 500         # Steps take per game
reqScore = 50           # Minimum score for game to be considered a success
initialGames = 10000    # Number of games played

def initialPopulation():
  # Array with saved training data
  trainingData = []

  # Array with scores for all games
  scores = []
  # Array with scores of successful games
  acceptedScore = []

  for game in range(initialGames):
    # Complete score for current game
    score = 0
    # List of all actions take for each step in the current game
    gameMemory = []
    # Observation recorded on previous game step
    prevObs = []
    for _ in range(goalSteps):
      # env.render()
      # Defining random action, either 0 or 1 representing movement of cart to either left or right
      action = random.randrange(0, 2)
      # Taking action for each game step
      observation, reward, done, info = env.step(action)

      # If there is an observation from the previous step -> save it to gameMemory
      if len(prevObs) > 0:
        gameMemory.append([prevObs, action])

      # Make current observation next step's previous observation
      prevObs = observation
      # Add reward to current game score
      score += reward

      if done:
        break
      
      if score >= reqScore:
        # Appending score to 'acceptedScore' if it is bigger than the minimum required score
        acceptedScore.append(score)
        for data in gameMemory:
          # If movement goes into one direction -> encode as 0,1
          if data[1] == 1:
            output = [0,1]
          # If movement goes into the other direction -> encode as 0,1
          elif data[1] == 0:
            output = [1,0]
          
          # Appending step from 'gameMemory' to 'trainingData' if it is bigger than the minimum required score
          # These are the examples the model will later learn from
          trainingData.append([data[0], output])
      
    env.reset()
    # Append score for current game
    scores.append(score)

  trainSave = np.array(trainingData)
  np.save('Data/trainSaved.npy', trainSave)

  print(f'\nAverage accepted score: {mean(acceptedScore)}\n')

initialPopulation()