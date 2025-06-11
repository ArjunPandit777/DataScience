import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

path = "C:\Datascience\Dataset\diabetes_dataset.csv"

df =  pd.read_csv(path)
print(df.head())
print(df.info())

X = df[['age','hypertension','heart_disease','bmi','HbA1c_level','blood_glucose_level',]]
y = df['diabetes']

print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model= LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

error = mean_squared_error(y_test,y_pred)      # [actual(observed)-predicted]square/sum of numbers = mean square error
print("Mean Squared Error:")

plt.scatter(X_test['age'],y_test,color='red')
plt.scatter(X_test['age'],y_pred,color='blue')



plt.xlim(60,80)
plt.ylim(0,1)
    
plt.xlabel('age')
plt.ylabel('diabetes')
plt.title('Diabetes Prediction')
plt.legend(['Actual','Predicted'])
plt.show()
plt.figure



# 


