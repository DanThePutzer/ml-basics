import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing, model_selection
import importlib

# Importing function to handle non-numerical data from previous file
handleData = importlib.import_module('35-Handling_Non-Numerical_Data')

# Loading data from excel file and handling non-numerical data
titanicData = pd.read_excel('Data/titanic.xls')
titanicData.fillna(0, inplace=True)
titanicData = handleData.handleNonNumericalData(titanicData)

# Defining features 'X' and label 'y'
X = np.array(titanicData.drop(['survived', 'body', 'name'], axis=1).astype(float))
y = np.array(titanicData['survived'])

# Scaling feature values
# Makes a HUGE difference, try without it
X = preprocessing.scale(X)

# Defining classifier
clf = KMeans(n_clusters=2)
# Training classifier
clf.fit(X)

correct = 0
for i in range(len(X)):
  # Turning numpy array into 1D matrix for classifier using reshape
  predictionSet = np.array(X[i].astype(float))
  predictionSet = predictionSet.reshape(-1, len(predictionSet))
  # Making prediction
  prediction = clf.predict(predictionSet)

  if prediction == y[i]:
    correct += 1

# Calculating accuracy
accuracy = correct/len(X)
print(accuracy)