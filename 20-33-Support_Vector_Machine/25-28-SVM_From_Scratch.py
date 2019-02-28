import numpy as np
import matplotlib.pyplot as plt

# Defining simple dataset
#   1 -> positive examples
#   -1 -> negative examples
dataset = {
  -1: np.array([
    [1,7],
    [2,8],
    [3,8],
    [2,7]
  ]),
  1: np.array([
    [5,1],
    [6,-1],
    [7,3]
  ])
}

# Defining SVM class
class SupportVectorMachine:
  def __init__(self, visualization=True):
    # Assigning instance-specific value of 'visualization' in __init__ with self
    # Have to learn more about python classes and 'self'
    self.visualization = visualization
    self.colors = {1: 'r', -1: 'b'}
    # Initializng matplotlib plot if visualization is True
    if self.visualization:
      self.fig = plt.figure()
      self.ax = self.fig.add_subplot(1,1,1)

  # Training SVM
  def fit(self, data):
    self.data = data

    # Dictionary with all magnitudes of y as keys and the respective points that produce them
    # { ||w||: [x,y] }
    optDict = {}
    # Transforms will be applied to w: Calculating the magnitude eliminates negative sign, but it still matters for calculating the dot product, therefore need to try all variants
    transforms = [[1,1], [1,-1], [-1,1], [-1,-1]]

    # Adding all data to one big list
    allData = []
    [[[allData.append(feature) for feature in featureset] for featureset in self.data[yi]] for yi in self.data]
    # Getting maximum and minimum values
    self.featureMax = max(allData)
    self.featureMin = min(allData)

    # Clearing 'allData' from memory
    allData = None

    # Defining step sizes for gradient descent
    stepSizes = [
      self.featureMax * 0.1,
      self.featureMax * 0.01,
      self.featureMax * 0.001,
      self.featureMax * 0.0001,
      self.featureMax * 0.00001
    ]

    # Very expensive, the higher the bigger the range for b
    b_RangeMultiple = 2
    # The higher the bigger the step size
    b_Multiple = 5

    # First values for vector w
    latestOptimum = self.featureMax * 10

    # Start optimizing step by step (Gradient Descent)
    
    # Within a certain range given below using np.arange we try to find the local minimum given a certain step size
    # Once we've tried all intervals within the given rang with one stepsize, we set 'w' and 'b' to the best ones and switch to the next step size
    for step in stepSizes:
      w = np.array([latestOptimum, latestOptimum])
      print(w)
      # As long as this is False -> Haven't reached optimum yet
      optimized = False
      while not optimized:
        # np.arange(start value, stop value, step size)
        for b in np.arange(-1*(self.featureMax * b_RangeMultiple), self.featureMax * b_RangeMultiple, step * b_Multiple):
          # Go through all possible sign variatons of vector w
          for transformation in transforms:
            wT = w * transformation
            foundOption = True

            # Filtering out 'b's and 'w's that yield values below 1 for the support vector
            for yi in self.data:
              for xi in self.data[yi]:
                if not yi * (np.dot(wT, xi) + b) >= 1:
                  foundOption = False

            # Adding transformed 'w's and 'b's to optDict 
            if foundOption:
              # linalg.norm is the magnitude
              optDict[np.linalg.norm(wT)] = [wT, b]
        
        # Getting a negative value from (w - step) calculated in previous iteration means that 'w' is smaller than step size and can't be optimized further using this step size
        if w[0] < 0:
          optimized = True
          print('Optimized a step')
        # If subtracting one step from 'w' didn't make it smaller than 0 we haven't reached the best possible value yet
        else:
          w = w - step

      # Sorting magnitude dictionary and picking lowest (-> best) value
      norms = sorted([n for n in optDict])
      optChoice = optDict[norms[0]]

      # Setting 'w' and 'b' to new best values
      self.w = optChoice[0]
      self.b = optChoice[1]
      latestOptimum = optChoice[0][0] + step * 2
      

  # Predicting Stuff
  def predict(self, features):
    # Prediction is based on sign(x • w + b)
    classification = np.sign(np.dot(np.array(features), self.w) + self.b)

    if classification != 0 and self.visualization:
      self.ax.scatter(features[0], features[1], s = 100, marker = '*', c = self.colors[classification])
    return classification

  # Function to visualize results
  def visualize(self):
    [[self.ax.scatter(x[0], x[1], s = 60, c = self.colors[i]) for x in dataset[i]] for i in dataset]

    # Draw hyperplanes
    def hyperplane(x, w, b, v):
      return (-w[0] * x - b + v) / w[1]

    datarange = (self.featureMin * 0.9, self.featureMax * 1.1)
    hypXMin = datarange[0]
    hypXMax = datarange[1]

    # print(hypXMin, self.w, self.b)
    # print(-self.w)

    # Positive support vector hyperplane
    # w•x+b = 1
    pos1 = hyperplane(hypXMin, self.w, self.b, 1)
    pos2 = hyperplane(hypXMax, self.w, self.b, 1)
    self.ax.plot([hypXMin, hypXMax], [pos1, pos2], 'k')

    # Negative support vector hyperplane
    # w•x+b = -1
    neg1 = hyperplane(hypXMin, self.w, self.b, -1)
    neg2 = hyperplane(hypXMax, self.w, self.b, -1)
    self.ax.plot([hypXMin, hypXMax], [neg1, neg2], 'k')

    # Decision boundary
    # w•x+b = 0
    db1 = hyperplane(hypXMin, self.w, self.b, 0)
    db2 = hyperplane(hypXMax, self.w, self.b, 0)
    self.ax.plot([hypXMin, hypXMax], [db1, db2], 'y--')

    plt.show()


# Trying classification
svm = SupportVectorMachine()
svm.fit(data = dataset)

# Defining prediction set
predictionset = [
  [0,10],
  [1,3],
  [3,4],
  [3,5],
  [5,5],
  [5,6],
  [6,-5],
  [5,8]
]

# Making predictions
for point in predictionset:
  svm.predict(point)

# Checking results of yi * (np.dot(wT, xi) + b)
# Should be as close to 1 as possible
[[print(xi,': ', yi * (np.dot(svm.w, xi) + svm.b)) for xi in dataset[yi]] for yi in dataset]
 
# Visualizing the whole thing
svm.visualize()