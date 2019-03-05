from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets.samples_generator import make_blobs

# - - - - Bulk copied from 41-Mean_Shift_From_Scratch.py - - - - 
# Improving it with dynamic bandwith selection

# Generating simple dataset
# centers = [[5,2], [9,12], [1,9]]

X, _ = make_blobs(n_samples=100, centers=6, cluster_std=1)
# X = np.array([[1, 2],
#               [1.5, 1.8],
#               [5, 8 ],
#               [8, 8],
#               [1, 0.6],
#               [9,11],
#               [8,2],
#               [10,2],
#               [9,3],])

# Colors for graphs
colors = 10*['y', 'c', 'm', 'g', 'r', 'b', 'k']

# Creating Mean Shift class
class MeanShift:
  def __init__(self, radius=None, radiusNormStep=100, radiusTweak=1):
    self.radius = radius
    self.radiusNormStep = radiusNormStep
    self.radiusTweak = radiusTweak

  # Training function
  def fit(self, data):

    # Checking if user set radius manually
    if self.radius == None:
      # Calculating center of all datapoints
      masterCentroid = np.average(data, axis=0)
      # Calculating distance to masterCentroid
      masterNorm = np.linalg.norm(masterCentroid)
      # Calculating new radius
      self.radius = masterNorm / (self.radiusNormStep * self.radiusTweak)
      print(self.radius)

    # Dictionary for centeroids
    centroids = {}
    # Looping over data
    for i in range(len(data)):
      centroids[i] = data[i]

    # Defining weights for dynamic bandwith selection
      # [::-1] -> Reverses list
    weights = [i for i in range(self.radiusNormStep)][::-1]

    while True:

      newCentroids = []
      for i in centroids:
        # List of points within bandwidth of current datapoint
        inBandwidth = []
        # List of weights for each featureset in 'inBandwidth'
        weightList = []
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
          if weightIndex > self.radiusNormStep - 1:
            weightIndex = self.radiusNormStep - 1

          # Appending current datapoint to 'inBandwidth'
          inBandwidth.append(datapoint)
          # Appending current datapoint's step distance from centeroid to 'weightList'
          weightList.append(weights[weightIndex])
        
        # Calculating new centeroids of clusters by taking average of all points within bandwidth and appending to 'newCenteroids' array
        newCentroid = np.average(inBandwidth, axis=0, weights=weightList)
        # print(newCentroid)
        newCentroids.append(tuple(newCentroid))
      # print('----')
      
      # Getting new unique centeroids -> eliminates duplicates and takes step towards convergence by unifying clusters with identical centeroids
      # Using set() to get uniques and sorted() to sort them by size
      uniques = sorted(list(set(newCentroids)))

      # Implementing method to eliminate centeroids which aren't identical, but very close to each other (beyond tolerance)
      # Comparing each centeroid to each other centeroid
      popPoints = []

      for i in uniques:
        # Skipping points already marked for popping
        if i in popPoints:
          break

        for j in uniques:
          # If they are the same, no need for action because already handled above with set()
          if i == j:
            pass
          # If distance between two centeroids is smaller than radius and point isn't already in 'popPoints' -> Within one step of each other -> Need to be converged -> Add to list of points to pop
          elif np.linalg.norm(np.array(i) - np.array(j)) <= self.radius and j not in popPoints:
            popPoints.append(j)
      

      # Removing points beyond tolerance
      for i in popPoints:
        # print(i)
        try:
          uniques.remove(i)
        except:
          pass


      # Saving previous centeroids for next run and replacing previous centeroids with newly determined ones
      prevCenteroids = dict(centroids)
      centroids = {}
      for i in range(len(uniques)):
        centroids[i] = np.array(uniques[i])

      optimized = True

      for i in centroids:
        if not np.array_equal(prevCenteroids[i], centroids[i]):
          optimized = False

      if optimized:
        break

    self.centroids = centroids

    # Classifying datapoints
    self.classifications = {}

    # Creating empty lists for each datapoint for each cluster
    for i in range(len(self.centroids)):
      self.classifications[i] = []

    # Classifying datapoints
    for datapoint in data:
      # Calculating distance for between datapoint and each centroid
      distances = [np.linalg.norm(datapoint - self.centroids[centroid]) for centroid in self.centroids]
      # Class of datapoint is class of centroid closest to it
      classification = distances.index(min(distances))
      # Appending classifications to list of closest centroid
      self.classifications[classification].append(datapoint)

  # Predicting function
  def predict(self, data):
    # Calculating distance for between datapoint and each centroid
    distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
    # Class of datapoint is class of centroid closest to it
    return distances.index(min(distances))


# Defining and training classifier
clf = MeanShift(radiusNormStep=50, radiusTweak=2)
clf.fit(X)

print(f'\n{len(clf.centroids)} Clusters Detected\n')

# Plotting dataset
for cluster in clf.classifications:
  for point in clf.classifications[cluster]:
    plt.scatter(point[0], point[1], color=colors[cluster], marker='o', s=30)

# Plotting centeroids
for c in clf.centroids:
  plt.scatter(clf.centroids[c][0], clf.centroids[c][1], marker='*', color='k', s=100)

plt.show()