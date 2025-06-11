import numpy as np

arr = np.array([1,2,3,4,5])

arr2 = arr.copy()
print(arr2)

arr2[1] = 100
print(arr)
print(arr2)

arr3 = arr.view()
print(arr3)

arr3[1] = 200
print(arr)
print(arr3)