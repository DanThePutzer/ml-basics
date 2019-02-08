import pandas as pd
import quandl

# Quandl API key
apiKey = 'yPnE7yZqyJzdxcz1pVeh'

# Fetching data from Quandl and loading it into dataframe
google = quandl.get('WIKI/GOOGL', authtoken=apiKey)
# Only keeping relevant columns
google = google[['Adj. Open', 'Adj. Close', 'Adj. High', 'Adj. Low', 'Adj. Volume']]


# Defining new, more useful features
google['HL Pct'] = (google['Adj. High'] - google['Adj. Low']) / google['Adj. Low'] * 100
google['Pct Change'] = (google['Adj. Open'] - google['Adj. Close']) / google['Adj. Close'] * 100

# Creating dataframe with final features
google = google[['Adj. Close', 'HL Pct', 'Pct Change', 'Adj. Volume']]

# Exporting data as pickle
google.to_pickle('Data/GoogleFeatures.pickle')