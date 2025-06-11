# Supervised Learning: Trains models on labeled data to predict or classify new, unseen data.
"""
age = [18, 19, 20, 21, 22]
salary = [30000, 35000, 40000, 45000, 50000]

45 => ?

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# age
X = np.array([18, 19, 20, 21, 22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50])
# salary
y = np.array([30000, 35000, 40000, 45000, 50000,55000,60000,65000,70000,75000,80000,85000,90000,95000,100000,105000,110000,115000,120000,125000,130000,135000,140000,145000,150000,155000,160000,165000,170000,175000,180000,185000,190000])

# add noise into the data
y = y + np.random.normal(0, 10000, len(y))  # 5000


X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(X_train.reshape(-1,1), y_train)
y_pred = model.predict(X_test.reshape(-1,1))
print(y_pred)
print(y_test)

plt.scatter(X_train,y_train,color="red")
plt.plot(X_test,y_pred,color="blue")
plt.show()