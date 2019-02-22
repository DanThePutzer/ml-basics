import numpy as np
import matplotlib.pyplot as plt

# Defining simple dataset
#   1 -> positive examples
#   -1 -> negative examples
dataset = {
  1: np.array([
    [1,7],
    [2,8],
    [3,8]
  ]),
  -1: np.array([
    [5,1],
    [6,-1],
    [7,3]
  ])
}

# Defining SVM class
class SupportVectorMachine:
  def __init__(self, visualization=True):
    # Assigning instance-specific value of 'visualization' in __init__ with self
    # Have to learn more about python classes and 'self'
    self.visualization = visualization
    self.colors = {1: 'r', -1: 'b'}
    # Initializng matplotlib plot if visualization is True
    if self.visualization:
      self.fig = plt.figure()
      self.ax = self.fig.add_subplot(1,1,1)

  # Training SVM
  def fit(self, data):
    self.data = data

    # Dictionary with all magnitudes of y as keys and the respective points that produce them
    # { ||w||: [x,y] }
    optDict = {}
    # Transforms will be applied to w: Calculating the magnitude eliminates negative sign, but it still matters for calculating the dot product, therefore need to try all variants
    transforms = [[1,1], [1,-1], [-1,1], [-1,-1]]

    # Adding all data to one big list
    allData = []
    [[[allData.append(feature) for feature in featureset] for featureset in self.data[yi]] for yi in self.data]
    # Getting maximum and minimum values
    self.featureMax = max(allData)
    self.featureMin = min(allData)

    # Clearing 'allData' from memory
    allData = None

    # Defining step sizes for gradient descent
    stepSizes = [
      self.featureMax * 0.1,
      self.featureMax * 0.01,
      self.featureMax * 0.001
    ]

    b_RangeMultiple = 5
    b_Multiple = 5

    # First values for vector w
    latestOptimum = self.featureMax * 10

    # Start optimizing step by step (Gradient Descent)
    for step in stepSizes:
      w = np.array([latestOptimum, latestOptimum])
      # As long as this is False -> Haven't reached optimum yet
      optimized = False
      while not optimized:
        


  # Predicting Stuff
  def predict(self, features):
    # Prediction is based on sign(x • w + b)
    classification = np.sign(np.dot(np.array(features), self.w) + self.b)

    return classification