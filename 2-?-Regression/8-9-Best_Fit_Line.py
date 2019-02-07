from statistics import mean
import numpy as np
import matplotlib.pyplot as plt

# Data
# Need numpy to actually create an array, native python does not have arrays, only lists
xData = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
yData = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)

xToPredict = np.array([10,12,7], dtype=np.float64)

# - Formulas used from 7-How_Regression_Works.txt -

# Computing y-intercept b
def intercept(x, y, m):
  b = mean(y) - m * mean(x)
  return b

# Computing slope m
def slope(x, y):
  m = (mean(x) * mean(y) - mean(x * y)) / (mean(x)**2 - mean(x**2))
  return m

# Computing best line fit
def bestFitLine(x, y):
  m = slope(x, y)
  b = intercept(x, y, m)
  return m, b

# Defining slope m and intercept b
m,b = bestFitLine(xData, yData)

print(f"\nSlope: {m}\ny-Intercept: {b}\n")

# Calculating y-points for regression line
# One line for-loop
regressionLine = [(m*x)+b for x in xData]


# Some simple predictions
predictionPoints = [(m*x)+b for x in xToPredict]

# Drawing training data and regression line
plt.scatter(xData, yData)

# Drawing predicted data and regression line
plt.scatter(xToPredict, predictionPoints)

# xData provides x-coordinates, regressionLine provides y-coordinates for line a point x
# Plotting without xData would shift line to start from x = 0
plt.plot(xData, regressionLine)
plt.show()

