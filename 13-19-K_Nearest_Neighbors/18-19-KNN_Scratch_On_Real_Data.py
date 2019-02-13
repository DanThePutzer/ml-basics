import numpy as np
import pandas as pd
from math import sqrt
import warnings
from collections import Counter
import random

# KNN function from 16-17-KNN_From_Scratch.py
def kNearestNeighbors(data, predict, k=3):
  # Warn if amount of used neighbors k is smaller than amountof different classes
  if len(data) >= k:
    warnings.warn('k is lower than total classes')
  
  # Initialize list of results
  voteResult = []
  distances = []

  # For each group in the dataset
  for group in data:
    # For each feature of a specific datapoint
    for features in data[group]:
      # Shorter numpy equivalent of what has been done in 15-Euclidian_Distance.py
      # Works for any number of dimensions without need for hardcoding
      euclDist = np.linalg.norm(np.array(features) - np.array(predict))
      distances.append([euclDist, group])

  # Grabs first k items of distances list sorted from smalles to biggest distance
  votes = [i[1] for i in sorted(distances)[:k]]
  # Find most common class among k closest datapoints and return class
  # Counter(-list-).most_common(number of most common items wanted)
  #   -> Returns list of tuples:
  #         tuple[0] -> Most common item
  #         tuple[1] -> How often most common item appears
  voteResult = Counter(votes).most_common(1)[0][0]
  confidence =  Counter(votes).most_common(1)[0][1] / k

  return voteResult, confidence

# Loading and preparing data
cancerData = pd.read_csv('Data/BreastCancer.csv')
cancerData.replace('?', -99999, inplace=True)
# Dropping 'id' column because it is a useless feature
cancerData.drop(['id'], axis=1, inplace=True)

# Function to split data into training and testing set (equiv. to model_selection.train_test_split in Sklearn)
def splitTrainTest(data, label, testSize):
  labels = data.groupby(label).nunique().index.values
  # Making sure all data consists of floats and not strings with .astype(float)
  # .values.tolist() coverts dataframe to a list of lists
  fullData = data.astype(float).values.tolist()
  # Shuffle data to eliminate any potential bias resulting from ordered data during training
  random.shuffle(fullData)

  # Generating dictionaries with labels in 'labels' as keys and empty lists as values
  # dict = {el:value for el in -list with keys-}
  trainSet = {el:[] for el in labels}
  testSet = {el:[] for el in labels}

  # Appending data to dictionary according to 'class'
  trainData = fullData[:-int(testSize*len(fullData))]
  testData = fullData[-int(testSize*len(fullData)):]

  # One line for-loops
  # Getting value of last column (which is the label column) of row 'i' in 'trainData'
  # to determine the key it needs to be appended to in the 'trainSet' dictionary and 
  # appending all values of row 'i' to 'trainSet' dictionary under appropriate key
  [trainSet[i[-1]].append(i[:-1]) for i in trainData]
  [testSet[i[-1]].append(i[:-1]) for i in testData]

  return trainSet, testSet

# Splitting data into train and test sets using function from above
trainSet, testSet = splitTrainTest(cancerData, 'class', 0.4)

# Metrics to track performance
total = 0
correct = 0

# Training classifier
# Going over each datapoint for each key in trainSet
for key in trainSet:
  for datapoint in trainSet[key]:
    # Running KNN for every datapoint in trainSet
    vote, confidence = kNearestNeighbors(trainSet, datapoint, 5)

    # Updating metrics
    if vote == key:
      correct += 1
    else:
      print(confidence)
    total += 1

print(f"\n Total: {total} \n Correct: {correct} \n\n Accuracy: {correct/total}\n")




