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
    # Learn more about python classes and 'self'
    self.visualization = visualization
    self.colors = {1: 'r', -1: 'b'}