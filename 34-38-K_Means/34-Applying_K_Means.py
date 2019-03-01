from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

# Creating simple dataset
X = np.array([
  [1,2],
  [1.5,1.8],
  [5,8],
  [8,8],
  [9,11]
])

# Defining and fitting classifier
clf = KMeans(n_clusters=2)
clf.fit(X)

# Getting some metrics and data from the classifier
centeroids = clf.cluster_centers_
labels = clf.labels_    # Returns what would be equal to 'y' in previous examples, label for each featureset in 'X' with same index

# Defining some colors for graph
colors = ["g.", "c.", "r.", "b.", "k."]

# Plotting classified dataset
[plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10) for i in range(len(X))]
plt.scatter(centeroids[:,0], centeroids[:,1], marker='*', s=120)
plt.show()