import numpy as np
from math import sqrt
import warnings
import matplotlib.pyplot as plt
from collections import Counter

# Defining dataset with two different classes (clusters)
dataset = {
  'b': [[1,2], [2,3], [3,1]],
  'r': [[6,5], [7,7], [8,6]]
}

# Defining new datapoints to classify
newFeatures = [[5,7], [3,2], [4,4]]

# Scatter plot dataset
# for i in dataset:
#   for j in dataset[i]:
#     plt.scatter(j[0], j[1], s=100, color=i)

# One line version of above loops
[ [ plt.scatter(j[0], j[1], s=100, color=i) for j in dataset[i] ] for i in dataset ]

# KNN function
def kNearestNeighbors(data, predict, k=3):
  # Warn if amount of used neighbors k is smaller than amountof different classes
  if len(data) >= k:
    warnings.warn('k is lower than total classes')
  
  # Initialize list of results
  voteResult = []

  # Added for loop to enable predictions on an array of ne datapoints
  for new in predict:

    distances = []

    # For each group in the dataset
    for group in data:
      # For each feature of a specific datapoint
      for features in data[group]:
        # Shorter numpy equivalent of what has been done in 15-Euclidian_Distance.py
        # Works for any number of dimensions without need for hardcoding
        euclDist = np.linalg.norm(np.array(features) - np.array(new))
        distances.append([euclDist, group])

    # Grabs first k items of distances list sorted from smalles to biggest distance
    votes = [i[1] for i in sorted(distances)[:k]]
    # Find most common class among k closest datapoints and return class
    # Counter(-list-).most_common(number of most common items wanted)
    #   -> Returns list of tuples
    voteResult.append(Counter(votes).most_common(1)[0][0])

  return voteResult

# Getting result
result = kNearestNeighbors(dataset, newFeatures, k=3)
print(result)

# Scatter to-predict features
# Using 'enumerate' function to get index of prediction and assign proper color to plotter point
[plt.scatter(new[0], new[1], s=300, c=result[index]) for index, new in enumerate(newFeatures)]
plt.show()
