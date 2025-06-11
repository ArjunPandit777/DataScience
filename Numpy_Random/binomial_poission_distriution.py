import numpy as np
# binomial distribution  # Discrete Distribution.
# poisson distribution  # Discrete Distribution.

numbers = np.random.binomial(n=100, p=0.5, size=10)  # probability of success
print(numbers)

numbersNew = np.random.poisson(lam=5, size=10)
print(numbersNew)

x = np.random.uniform(size=(2, 3))
print(x)