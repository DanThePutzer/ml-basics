import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection, neighbors

# Importing Data
cancerData = pd.read_csv('Data/BreastCancer.csv')
cancerData.drop('id', axis=1, inplace=True)

# Converting NaN (denoted as '?') to outliers
cancerData.replace('?', -99999, inplace=True)

print(cancerData.head(10))

# Defining X and y as numpy arrays
# Axis Usage:
#   0 - drops from index (rows)
#   1 - drops from columns
X = np.array(cancerData.drop('class', axis=1))
y = np.array(cancerData['class'])

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# Defining KNN classifier
clf = neighbors.KNeighborsClassifier()

# Training classifier
clf.fit(X_train, y_train)

# Testing classifier
# Try training and testing WITHOUT dropping 'id' column to see how devastating irrelevant features can be
accuracy = clf.score(X_test, y_test)
print(f"\nTest Accuracy: {accuracy}")

# Predicting Stuff
X_predict = np.array([4,2,1,1,1,2,3,2,1]).reshape(1, -1)
prediction = clf.predict(X_predict)
print(f"Prediction: {prediction}\n")

# - - - - NumPy .reshape(row, col): - - - -
#   row = some integer -> Reshaped array will have specified number of rows
#   column = some integer -> Reshaped array will have specified number of columns
#   row or col = -1 -> NumPy figures out remaining dimensions of array
#     -> Cannot have row and col be -1, one of the two has to be specified

# 1D array has to be reshaped to (1, -1) = 1 row and number of cols determined by NumPy