# tod do

import numpy as np 
mainArray = np.array([1,2,3,4,5,6,7,8,9,10])

newArray = np.array_split(mainArray,5)
print(newArray) # 2 * 5

main2dArray = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]])  # 5 * 3

data = np.array_split(main2dArray,2)
print(data)

# horizontalsplit and verticalsplit

print("start")

arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12],[13, 14, 15, 1]]) # 3 * 4

hsplit = np.hsplit(arr,2)
print(hsplit)

vsplit = np.vsplit(arr,2)
print(vsplit)
