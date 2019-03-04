import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
from sklearn import preprocessing, model_selection

# - Function copied from 35-Handling_Non-Numerical_Data.py

# Function to handle non-numerical data
def handleNonNumericalData(df):
  columns = df.columns.values

  # Looping over each column in dataframe
  for column in columns:
    # textDigitValues contains key - value pairs
      # Keys are original name of column entry
      # Values are the integers replacing them
    textDigitValues = {}

    # Returns int value for a certain key textDigitValues
    def convertToInt(val):
      return textDigitValues[val]

    # For each column we check if it is an int or float
    if df[column].dtype != np.int64 and df[column].dtype != np.float64:
      # Converting column entries to list
      columnContent = df[column].values.tolist()
      # Find all unique elements using set()
      uniqueElements = set(columnContent)

      count = 0
      # For each unique element...
      for uniqueElement in uniqueElements:
        # ...if it isn't part of textDigitValues yet... 
        if uniqueElement not in textDigitValues:
          # ... we add a new entry to textDigitValues with it's name and an assigned integer unique to that name
          textDigitValues[uniqueElement] = count
          count += 1
      
      # Mapping the ints replacing the text to the column we started out with
      df[column] = list(map(convertToInt, df[column]))

  return df


# - Parts copied from 36-K_Means_Titanic.py

# Loading data from excel file and handling non-numerical data
titanicData = pd.read_excel('Data/titanic.xls')
# Making deep copy of original data
originalTitanicData = pd.DataFrame.copy(titanicData)
titanicData.fillna(0, inplace=True)
titanicData = handleNonNumericalData(titanicData)

# Defining features 'X' and label 'y'
X = np.array(titanicData.drop(['survived', 'body', 'name'], axis=1).astype(float))
y = np.array(titanicData['survived'])

# Scaling feature values
# Makes a HUGE difference, try without it
X = preprocessing.scale(X)

# Defining classifier
clf = MeanShift()
# Training classifier
clf.fit(X)

# Turning labels into a dataframe
clusterFrame = pd.DataFrame(clf.labels_)
clusterFrame.columns = ['Cluster']

# print(originalTitanicData.head())
# Adding labels column to original dataframe
originalTitanicData = pd.concat([originalTitanicData, clusterFrame], axis=1)

# Getting number of clusters MeanShift found
clusters = len(np.unique(clf.labels_))

# Calculating survival rate for different clusters
survivalRates = {}
for i in range(clusters):
  # Using a one line conditional statement
  # Creating temporary dataframe 'temp' which contains only featuresets where originaltitanicData['Cluster] equals current cluster i
  temp = originalTitanicData[ (originalTitanicData['Cluster'] == float(i)) ]
  # Calculating survival rate
  rate = len(temp[ (temp['survived'] == 1) ]) / len(temp['survived'])
  survivalRates[i] = rate

print(survivalRates)