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
accuracy = clf.score(X_test, y_test)

