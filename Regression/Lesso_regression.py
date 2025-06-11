# lesso regression
# when to use
# 1. when we don't have a linear relationship between the input and output variables
# 2. when we have a non-linear relationship between the input and output variables
# 3. when we have multiple input variables and we want to predict the output variable based on all the input variables

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder


path = "C:\Datascience\Dataset\CAR DETAILS FROM CAR DEKHO.csv"

df = pd.read_csv(path)
print(df.head())
print(df.info())

"""

Data columns (total 9 columns):
 #   Column         Non-Null Count  Dtype
---  ------         --------------  -----
 0   Car_Name       301 non-null    object    remove
 1   Year           301 non-null    int64
 2   Selling_Price  301 non-null    float64   # predict
 3   Present_Price  301 non-null    float64
 4   Kms_Driven     301 non-null    int64
 5   Fuel_Type      301 non-null    object
 6   Seller_Type    301 non-null    object
 7   Transmission   301 non-null    object
 8   Owner          301 non-null    int64

"""

# covert int to float
df['year'] = df['year'].astype(float)
df['kms_Driven'] = df['kms_Driven'].astype(float)
df['Owner'] = df['Owner'].astype(float)
# label encoding

lc = LabelEncoder()
df['Fuel_Type'] = lc.fit_transform(df['Fuel_Type']).astype(float)
df['Seller_Type'] = lc.fit_transform(df['Seller_Type']).astype(float)
df['Transmission'] = lc.fit_transform(df['Transmission']).astype(float)

print(df.head())
print(df.info())


X = df[['year','Present_Price','kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner']]
Y = df['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

lasso = Lasso()
lasso.fit(X_train, y_train)

y_pred = lasso.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

# plot the data
plt.scatter(X_test['Present_Price'], y_test, color='blue', label='Actual')
plt.scatter(X_test['Present_Price'], y_pred, color='red', label='Predicted')
plt.title('Lasso Regression')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.show()