from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Creating simple dataset
X = np.array([
  [1,2],
  [1.5,1.8],
  [5,8],
  [8,8],
  [9,11],
  [12,3],
  [4,3],
  [2,6],
  [4,2],
  [11,11],
  [7,0.5],
  [0.5,8],
  [12,7],
  [4,14],
  [9,2],
  [11,1],
])

# Defining K Means Class
class K_Means:
  def __init__(self, k=2, tol=0.001, maxIter=300):
    self.k = k
    self.tol = tol
    self.maxIter = maxIter
    self.centeroidHistory = []

  # Training function
  def fit(self, data):
    # Dictionary containing k centeroids
    self.centeroids = {}

    # Shuffling data
    # np.random.shuffle shuffles list in place
    np.random.shuffle(data)
    # Choosing k centeroids from shuffled data
    for i in range(self.k):
      self.centeroids[i] = data[i]

    self.centeroidHistory.append([self.centeroids[0], self.centeroids[1]])

    for i in range(self.maxIter):
      # Has k lists, one for each of the k clusters
      self.clusters = {}

      # Create empty list for each of the possible k clusters
      for j in range(self.k):
        self.clusters[j] = []

      # Iterate over each featureset
      for featureset in data:
        # Calculate euclidian distance to each centeroid for each point
        distances = [np.linalg.norm(featureset - self.centeroids[centeroid]) for centeroid in self.centeroids]
        # Award class of closest centeroid
        classification = distances.index(min(distances))

        # Add featureset to its given classification in clusters dictionary
        self.clusters[classification].append(featureset)

      # Saving centeroid to compare for next iteration
      prevCenteroids = dict(self.centeroids)

      # Finding average of assigned classifications for each centeroid to adjust positions
        # For each cluster ( cluster in self.clusters )
        # set the centeroid ( self.centeroids[cluster] )
        # to the average of points ( self.clusters[cluster] ) belonging to the cluster
      for cluster in self.clusters:
        self.centeroids[cluster] = np.average(self.clusters[cluster], axis=0)
        self.centeroidHistory.append([self.centeroids[0], self.centeroids[1]])

      optimized = True

      # Comparing new centeroids to those from previous iteration
      for c in self.centeroids:
        prevC = prevCenteroids[c]
        newC = self.centeroids[c]

        # If new centeroids moved more than tolerance compared to previous centeroids -> not optimized yet
        # Only when adjustments become smaller than tolerance we are optimized
        if np.sum((newC - prevC) / prevC * 100) > self.tol:
          optimized = False

      if optimized:
        break


  # Predicting function
  def predict(self, data):
    # Calculating distance to all the different clusters
    distances = [np.linalg.norm(data - self.centeroids[centeroid]) for centeroid in self.centeroids]
    # Finding name of closest one
    classification = distances.index(min(distances))

    return classification



# Defining some colors for graph
colors = ["g", "c", "r", "b"]


clf = K_Means()
clf.fit(X)

for centeroid in clf.centeroids:
  plt.scatter(clf.centeroids[centeroid][0], clf.centeroids[centeroid][1], marker='*', s=70, c='k')

for cluster in clf.clusters:
  color = colors[cluster]
  for datapoint in clf.clusters[cluster]:
    plt.scatter(datapoint[0], datapoint[1], marker='o', s=30, c=color)

plt.show()