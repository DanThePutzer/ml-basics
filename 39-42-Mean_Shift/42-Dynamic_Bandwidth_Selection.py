from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets.samples_generator import make_blobs

# - - - - Bulk copied from 41-Mean_Shift_From_Scratch.py - - - - 
# Improving it with dynamic bandwith selection

# Generating simple dataset
centers = [[5,2], [9,12], [1,9]]

X, _ = make_blobs(n_samples=100, centers=centers, cluster_std=1.5)

# Colors for graphs
colors = ['y', 'c', 'm', 'g', 'r', 'b', 'k']

# Creating Mean Shift class
class MeanShift:
  def __init__(self, radius=None, radiusNormStep=100):
    self.radius = radius
    self.radiusNormStep = radiusNormStep

  # Training function
  def fit(self, data):

    # Checking if user set radius manually
    if self.radius == None:
      # Calculating center of all datapoints
      masterCentroid = np.average(data, axis=0)
      # Calculating distance to masterCentroid
      masterNorm = np.linalg.norm(masterCentroid)
      # Calculating new radius
      self.radius = masterNorm / self.radiusNormStep

    # Dictionary for centeroids
    centroids = {}
    # Looping over data
    for i in range(len(data)):
      centroids[i] = data[i]

    # Defining weights for dynamic bandwith selection
      # [::-1] -> Reverses list
    weights = [i for i in range(self.radiusNormStep)][::-1]

    while True:
      # Collecting centeroids and their assigned points for colored plotting
      self.clusteredData = {}

      newCentroids = []
      for i in centroids:
        # Array of points within bandwidth of current datapoint
        inBandwidth = []
        
        # Looping over all datapoints and calculating distance and weight
        for datapoint in data:
          # Calculating distance between datapoint and clusters centroid
          distance = np.linalg.norm(datapoint-centroids[i])
          if distance == 0:
            distance = 0.00000001
          
          # Weight index tells us how much a datapoint should be weighted depending on how many radius steps away it is from the centeroid
            # Higher weight index -> lower weight because further away
          weightIndex = int(distance/self.radius)
          # Setting weight index to maximum steps if it goes beyond maximum steps
          if weightIndex > self.radiusNormStep:
            weightIndex = self.radiusNormStep - 1
        
        # Adding clusters and their respective points to 'clusteredData'
        self.clusteredData[i] = inBandwidth
        
        # Calculating new centeroids of clusters by taking average of all points within bandwidth and appending to 'newCenteroids' array
        newCentroid = np.average(inBandwidth, axis=0)
        newCentroids.append(tuple(newCentroid))
      
      # Getting new unique centeroids -> eliminates duplicates and takes step towards convergence by unifying clusters with identical centeroids
      # Using set() to get uniques and sorted() to sort them by size
      uniques = sorted(list(set(newCentroids)))

      # Saving previous centeroids for next run and replacing previous centeroids with newly determined ones
      prevCenteroids = dict(centroids)
      centroids = {}
      for i in range(len(uniques)):
        centroids[i] = np.array(uniques[i])

      optimized = True

      for i in centroids:
        if not np.array_equal(prevCenteroids[i], centroids[i]):
          optimized = False
        if not optimized:
          break

      if optimized:
        break

    self.centroids = centroids

  # Predicting function
  def predict(self, data):
    pass


# Defining and training classifier
clf = MeanShift()
clf.fit(X)

print(f'\n{len(clf.centroids)} Clusters Detected\n')

# Plotting dataset
# plt.scatter(X[:,0], X[:,1], s=30)
for cluster in clf.clusteredData:
  for point in clf.clusteredData[cluster]:
    plt.scatter(point[0], point[1], color=colors[cluster])

# Plotting centeroids
for c in clf.centroids:
  plt.scatter(clf.centroids[c][0], clf.centroids[c][1], marker='*', color='k', s=100)

plt.show()