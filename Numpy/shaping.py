# datatypes in array
# string
# int
# float
# complex
# bool


import numpy as np

arr = np.array([1,2,3,4,5])
print(arr.dtype)

arr2 = np.array([1.0,2.0,3.0,4.0,5.0])
print(arr2.dtype)

arr3 = np.array(['a','b','c','d','e'])  # ?
print(arr3.dtype)

arr4 = np.array([True, False, True])
print(arr4.dtype)

arr5 = np.array([1+2j, 2+3j, 3+4j])
print(arr5.dtype)

arr6 = np.array([1,2,0,4,5],dtype=bool)
print(arr6.dtype)
print(arr6)


# change datatype of an array

arr7 = np.array([1,2,3,4,5])
arr8 = arr7.astype("S")
print(arr8.dtype)