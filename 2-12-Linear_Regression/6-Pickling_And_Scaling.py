import pandas as pd
import numpy as np
import pickle
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt


# Importing model from pickle
pickle_in = open('Data/GoogleStockClassifier.pickle', 'rb')
clf = pickle.load(pickle_in)

# Importing data
google = pd.read_pickle('Data/GoogleFeatures&Label.pickle')
predictionSet = np.array(google.drop(['Label'], axis=1))
predictionSet = preprocessing.scale(predictionSet)
predictionSet = predictionSet[-35:]

# Predict
results = clf.predict(predictionSet)
print(results)
plt.plot(results)
plt.show()

