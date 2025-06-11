import numpy as np

arr = np.array([1,2,3,4,5])
for i in arr:
  print(i)

data = np.array([ [1,2,3] , [4,5,6] , [7,8,9] ]) # 3*3

for i in data:
  for j in i:
    print(j)

data3d = np.array([ [ [1,2,3] , [4,5,6] ] , [ [7,8,9] , [10,11,12] ] ]) # 2 * 2 * 3

for i in data3d:
  for j in i:
    for k in j:
      print(k)