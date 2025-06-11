# normal distribution (Gaussian Distribution)

# uniform distribution
# binomial distribution
# poisson distribution

# normal distribution
# 3 parameters: mean, standard deviation, and variance

# 5 4 5 4 5 4 5 4 3 5
# mean = 5
# (5-5)^2 + (4-5)^2 + (5-5)^2 + (4-5)^2 + (5-5)^2 + (4-5)^2 + (5-5)^2 + (4-5)^2 + (3-5)^2 + (5-5)^2/10

# mean = (5+4+5+4+5+4+5+4+3+5)/10
# varience = (5-5)^2 + (4-5)^2 + (5-5)^2 + (4-5)^2 + (5-5)^2 + (4-5)^2 + (5-5)^2 + (4-5)^2 + (3-5)^2 + (5-5)^2/10
# stadard deviation = sqrt(variance) = sqrt(0.5) = 0.7071067811865476

import numpy as np

arr = np.random.normal(10,5,(2,2))
print(arr)