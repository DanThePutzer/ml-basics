import pandas as pd
import numpy as np
import math
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression

# Reading data from pickle
google = pd.read_pickle('Data/GoogleFeatures&Label.pickle')
google.dropna(inplace=True)

print(google.head())

# Defining features (X) and label (y)
X = np.array(google.drop(['Label'], axis=1))
y = np.array(google['Label'])

# Using Sklearn's preprocessing module to scale X
X = preprocessing.scale(X)

# Checking if feature rows and label rows equal
print(f'\nX length: {len(X)} \ny length: {len(y)}\n')

# Create training and testing sets
# Train values before test values, X values before y values
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# Training and testing model
# Instead pf LinearRegression(), try svm.SVR()
# n_jobs: number of jobs run in parallel, -1 -> as many jobs as possible
clf = LinearRegression(n_jobs=20)
# Fit = training
clf.fit(X_train, y_train)
# Score = testing
accuracy = clf.score(X_test, y_test)

print(f'Obtained Accuracy: {accuracy}\n')