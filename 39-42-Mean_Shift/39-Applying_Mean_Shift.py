import numpy as np
from sklearn.cluster import MeanShift
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generating a dataset
centers = [[1,1,1], [5,5,5], [3,10,10]]
# make_blobs() generates  random data for clustering
  # - n_samples: Number of samples
  # - centers: Center points
  # - cluster_std: Standard deviation of clusters
X, _ = make_blobs(n_samples = 100, centers = centers, cluster_std = 1.25)


# Defining and training classifier
clf = MeanShift()
clf.fit(X)

# Getting some important data from classifier
clusterLabels = clf.labels_
clusterCenters = clf.cluster_centers_

clusterCount = len(np.unique(clf.labels_))
print(f'Number of clusters found: {clusterCount}')

# 3D plot of data
colors = ['y', 'c', 'm', 'g', 'r', 'b', 'k']
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scattering data
for i in range(len(X)):
  ax.scatter(X[i][0], X[i][1], X[i][2], c=colors[clusterLabels[i]], marker='o', s=20)

# Scattering cluster centers
ax.scatter(clusterCenters[:,0], clusterCenters[:,1], clusterCenters[:,2], marker='*', color='k', s=120, zorder=10)

plt.show()