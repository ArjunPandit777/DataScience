import numpy  as np

value = np.random.randint(100)  # [0,100]
print(value)

otherValue = np.random.rand() # [0,1]
print(otherValue)

# generate random value in an array

arr = np.random.randint(100, size=(2,2))
print(arr)

floatArr = np.random.rand(2,2)
print(floatArr)

myArr = np.array([1,2,3,4,5,6,7,8,9,10])
randomFromArray = np.random.choice(myArr)
print(randomFromArray)

myArr = np.random.choice(myArr, size=(2,2))
print(myArr)