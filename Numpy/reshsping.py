import numpy as np

arr = np.array([1,2,3,4,5,6])  # 3*2
print(arr)
print(arr.shape)


# reshape the array
arrNew1 = arr.reshape(1,6)  # indimint
arrNew2 = arr.reshape(3,2)  # indimint
print(arrNew1)
print(arrNew2)

arrNew3 = arr.reshape(3,2,-1)
print(arrNew3)

arr2d = np.array([ [1,2,3] , [4,5,6] , [7,8,9] ])  # 3*3
print(arr2d.shape)

arrNew4 = arr2d.reshape(-1)  # flatterning of and array
print(arrNew4)