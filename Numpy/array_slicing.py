import numpy as np

a = np.array(["banana","apple","mango","orange","grapes","kiwi","pineapple"])
print(a)
print(a[3:7]) # slicing

b = a[3:7]
print(b)

c = a[0:7:2]  # start : end-1 hota he : step
print(c)

d = a[4:]
print(d)

e = a[:4]
print(e)

f = a[::3] # only step
print(f)

# slicing 2d array
arr2d = np.array([ [1,2,3,4,5] , [6,7,8,9,10] ])
print(arr2d[0:3][0:3])