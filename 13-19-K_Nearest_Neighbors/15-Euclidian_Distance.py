# - - - - - Euclidian Distance - - - - -

# Formula = sqrt( [n sum() i = 1] * (q_i - p_i)**2 )

# Exp:
#  p = (1,3)
#  q = (2,5)

#  d = sqrt( (1 - 2)**2 + (3 - 5)**2 )

# - - Code - -
from math import sqrt
p = [1,3]
q = [2,5]

euclDist = sqrt( (p[0] - q[0])**2 + (p[1] - q[1])**2 )

print(f"\nEucl. Distance:{euclDist}\n'")