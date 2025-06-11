import pandas as pd

age = [21, 22, 23, 24, 25]
name = ["John", "Jane", "Jack", "Jill", "Joe"]

# Creating a Series using a list
s = pd.Series(age,index=["a", "b", "c", "d", "e"])
print(s)

s1 = pd.Series(name)
print(s1)


print(s[0])
print(s["e"])