import pandas as pd

# read csv file
df = pd.read_csv('/home/arpit-parekh/Downloads/archive(5)/movies.csv')
print(df.head())
print(df.info())


# df.fillna(130, inplace = True)  # replace all the null values with 130
# df['Gross'].fillna(130, inplace = True)  # replace gross column's null values with 130


# not possible to replace null values with mean value because of object datatype
# grossAverage = df['Gross'].mean()
# print(grossAverage)

# df['Gross'].fillna(grossAverage, inplace = True)  # replace gross column's null values with mean value

# df.dropna(inplace = True)
print(df.info())
print(df.head())

# df['Date'] = pd.to_datetime(df['Date'])  # format date column as datetime