import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing, model_selection

# Loading data from xls file
titanicData = pd.read_excel('Data/titanic.xls')
# Dropping irrelevant columns
titanicData.drop(['body', 'name'], axis=1, inplace=True)
titanicData.fillna(0, inplace=True)

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

# Applying function to deal with non-numerical data
titanicData = handleNonNumericalData(titanicData)
