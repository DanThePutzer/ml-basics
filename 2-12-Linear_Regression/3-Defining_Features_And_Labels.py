import pandas as pd
import math

# Reading in data from pickle
google = pd.read_pickle('Data/GoogleFeatures.pickle')

# Choosing column to use as label for training
forecastCol = 'Adj. Close'
# Fill NaN with some high number, will be treated as outliers
google.fillna(-99999, inplace=True)

# Determining how far into the future we want to predict (shift the data)
# 10% of the entire dataset for now
forecastOut = int(math.ceil(0.01*len(google)))

# Adding Label to dataframe
google['Label'] = google[forecastCol].shift(-forecastOut)
# google.dropna(inplace=True)

# Exporting data to pickle
google.to_pickle('Data/GoogleFeatures&Label.pickle')