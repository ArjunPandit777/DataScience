# joing two arrays using numpy

import numpy as np

arr1 = np.array([1,2,3,4,5])
arr2 = np.array([6,7,8,9,10])

arr3 = np.concatenate((arr1,arr2))
print(arr3)

arr2d1 = np.array([ [1,2] , [3,4] ]) # 2 * 2
arr2d2 = np.array([ [5,6] , [7,8] ]) # 2 * 2

# arr2d3 = np.concatenate((arr2d1,arr2d2))
arr2d3 = np.concatenate((arr2d1,arr2d2),axis=1)
print(arr2d3)  # 2 *4

print("Start")

# join two arrays using hstack, vstack , dstack
arr4 = np.hstack((arr2d1,arr2d2))  # same as axis=1
print(arr4)

print()

arr5 = np.vstack((arr2d1,arr2d2))
print(arr5)

print()

arr6 = np.dstack((arr2d1,arr2d2))
print(arr6)