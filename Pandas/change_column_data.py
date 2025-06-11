import pandas as pd

# read csv file
df = pd.read_csv('/home/arpit-parekh/Downloads/archive(5)/movies.csv')
print(df.head())
print(df.info())

df.loc[0,"RunTime"]= 122.0
print(df.head())

for x in df.index:
  if df.loc[x,"RunTime"] < 40:
    df.loc[x,"RunTime"] = 50

print(df.head())

for x in df.index:
  if df.loc[x,"RunTime"] < 60:
    df.drop(x, inplace=True)

print(df.head())

print(df.duplicated())

df.drop_duplicates(inplace = True)
print(df.head())