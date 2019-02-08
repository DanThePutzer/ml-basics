import pandas as pd
import numpy as np
import math, datetime, pickle
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LinearRegression

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')

# Import data
google = pd.read_pickle('Data/GoogleFeatures&Label.pickle')

# Forecast period
forecastOut = int(math.ceil(0.01*len(google)))

# Defining X and y
X = np.array(google.drop(['Label'], axis=1))
# Scaling X
X = preprocessing.scale(X)

# Rows up to first row with forecast label, will be used to predict against
X_lately = X[-forecastOut:]
# Dropping all rows up to the first forecast label, to avoid NaNs
X = X[:-forecastOut]

google.dropna(inplace=True)
y = np.array(google['Label'])

# Splitting Training and Test sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# Training and testing model
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)

# Save trained classifier to pickle for later use without need for training
with open('Data/GoogleStockClassifier.pickle', 'wb') as f:
  pickle.dump(clf, f)

# Test classifier
accuracy = clf.score(X_test, y_test)

# print(f"\Test accuracy: {accuracy}\n")

# Predicting Stuff
forecastSet = clf.predict(X_lately)

print(forecastSet)


# Plotting Forecasts
google['Forecast'] = np.nan

lastDate = google.iloc[-1].name
lastUnix = lastDate.timestamp()
futureUnix = lastUnix + 86400

# print(lastDate, lastUnix)

for i in forecastSet:
  newDate = datetime.datetime.fromtimestamp(futureUnix)
  futureUnix += 86400
  # loc: references index of dataframe
  # One line for-loop, adds NaNs to all columns because we don't have data except for Forecast column
  google.loc[newDate] = [np.nan for _ in range(len(google.columns)-1)] + [i]

google['Adj. Close'].tail(200).plot()
google['Forecast'].tail(200).plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
