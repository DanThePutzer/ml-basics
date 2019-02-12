import numpy as np
import pandas as pd
from math import sqrt
import warnings
import matplotlib.pyplot as plt
from collections import Counter

# Defining dataset with two different classes (clusters)
dataset = {
  'k': [[1,2], [2,3], [3,1]],
  'r': [[6,5], [7,7], [8,6]]
}

# Defining new datapoints to classify
newFeatures = [5,7]

# Scatter plot dataset
# for i in dataset:
#   for j in dataset[i]:
#     plt.scatter(j[0], j[1], s=100, color=i)

# One line version of above loops
[ [ plt.scatter(j[0], j[1], s=100, color=i) for j in dataset[i] ] for i in dataset ]

# Plotting to-predict features
plt.scatter(newFeatures[0], newFeatures[1])
plt.show()

# KNN function
def kNearestNeighbors(data, predict, k=3):
  # Warn if amount of used neighbors k is smaller than amountof different classes
  if len(data) >= k:
    warnings.warn('k is lower than total classes')

