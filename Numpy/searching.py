import numpy as np

arr = np.array([1,6,3,4,5,7,2])
x = np.where(arr==4)
print(x)

# where => index

y = np.where(arr%2==0)
print(y)

# search sorted binary search
# first sort the array then find the value
z = np.searchsorted(arr, 6)
print(z)

z1 = np.searchsorted(arr, 6, side="right")
print(z1)