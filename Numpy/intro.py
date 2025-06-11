import numpy as np

# list dynamic arrays
# arrays
# arrays are the group of elements of the same type
# numpy arrays are the group of elements of the same type

arr = np.array([1,2,3,4,5])
print(arr)

arr0 = np.array(5)
print(arr0)

arr1d = np.array([1,2,3,4,5])
print(arr1d)

arr2d = np.array([ [1,2] , [3,4] ])
print(arr2d)

# 1 => [1,2]
# 2 => [3,4]
# 3 => [5,6]
# 4 => [7,8]

arr3d = np.array([ [ [1,2] , [3,4] ] , [[5,6],[7,8] ] ])
print(arr3d)

# ndim => number of dimensions

print(arr0.ndim)
print(arr1d.ndim)
print(arr2d.ndim)
print(arr3d.ndim)


# myarray
# ndmin => number of dimensions

myarray = np.array([1,2,3,4,5,6],ndmin=2)  # 1 * 6
print(myarray)

another = np.array([1,2,3,4,5,6],ndmin=3) # 1 * 1 * 6
print(another)

mera = np.array([ [1,2],[3,4] ],ndmin=3)
print(mera)