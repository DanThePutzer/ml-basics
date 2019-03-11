import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter

# Defining lemmatizer
lem = WordNetLemmatizer()
lineCount = 1000000

# Function to create lexicon of all words in data
def createLexicon(pos, neg):
  lexicon = []
  for fi in [pos, neg]:
    # Opening files from array in 'read' mode
    with open(fi, 'r', encoding="utf-8") as f:
      # Reading file lines
      content = f.readlines()
      for l in content[:lineCount]:
        # Each line get tokenized -> Split into individual words
        allWords = word_tokenize(l.lower())
        # Adding words to general lexicon
        lexicon += list(allWords)

  # Lemmatizing lexicon
  lexicon = [ lem.lemmatize(word) for word in lexicon ]
  # Gives dictionary of words and how many times they appear
  wordCounts = Counter(lexicon)

  # Defining final lexicon cleansed of common non-sentimental words
  finalLexicon = []

  # Getting rid of very common words like 'the' etc which are not important for sentiment analysis and would only confuse the model
  for word in wordCounts:
    if 1000 > wordCounts[word] > 50:
      finalLexicon.append(word)

  print(len(finalLexicon))
  return finalLexicon

# Function to handle sample
def sampleHandling(sample, lexicon, classification):
  featureset = []
  # Opening sample file in 'read' mode
  with open(sample, 'r', encoding="utf-8") as f:
    # Reading file lines
    content = f.readlines()
    for l in content[:lineCount]:
      # Tokenizing each word in line
      current = word_tokenize(l.lower())
      # Lemmatizing all words in current array
      current = [ lem.lemmatize(word) for word in current ]
      # Creating features array with zeros as placeholders
      features = np.zeros(len(lexicon))
      for word in current:
        # If word in lexicon -> get index of word in lexicon and increase its value in features array to one indicating that the word exits in the current line of the file
        if word.lower() in lexicon:
          indexValue = lexicon.index(word.lower())
          features[indexValue] += 1
      features = list(features)
      featureset.append([features, classification])

  return featureset

# Function creating features and labels to train NN on
def createFeaturesAndLabels(pos, neg, test_size=0.1):
  # Creating lexicon with createLexicon function
  lexicon = createLexicon(pos, neg)

  # Getting array of features from datasets, combining featuresets into one and shuffling the hole thing -> Turning it into a numpy array
  features = []
  features += sampleHandling('Data/pos.txt', lexicon, [1,0])
  features += sampleHandling('Data/neg.txt', lexicon, [0,1])
  random.shuffle(features)
  features = np.array(features)

  # Defining test size for splitting data into training and testing sets
  testSize = int(test_size * len(features))

  # Dividing data into train and test sets
    # [:,0] -> 0th element of all rows
  trainX = list(features[:,0][:-testSize])
  trainY = list(features[:,1][:-testSize])

  testX = list(features[:,0][-testSize:])
  testY = list(features[:,1][-testSize:])

  return trainX, trainY, testX, testY

if __name__ == "__main__":
  trainX, trainY, testX, testY = createFeaturesAndLabels('Data/pos.txt', 'Data/neg.txt')
  with open('Data/SentimentSet.pickle', 'wb') as f:
    pickle.dump([trainX, trainY, testX, testY], f)
