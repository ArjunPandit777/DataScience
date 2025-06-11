# polinomial regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures #  Extra Feature
from sklearn.model_selection import train_test_split

# y = mx + c
# y = ax^2 + bx + c
# y = ax^3 + bx^2 + cx + d
# y = ax^4 + bx^3 + cx^2 + dx + e
# y = ax^5 + bx^4 + cx^3 + dx^2 + ex + f

X = np.random.rand(100, 1) # range 1 to 100
y =  15*X**3 + 20 * X**2 + 10*X + np.random.randn(100, 1)*3

poli_features = PolynomialFeatures(degree=3)
X = poli_features.fit_transform(X)

plt.ylabel('y')
plt.legend()
plt.show()





