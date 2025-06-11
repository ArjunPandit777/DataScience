import pandas as pd

csv = "/home/arpit-parekh/Downloads/customers-100.csv"

pd.options.display.max_rows = 9999   # to display more rows in output

df = pd.read_csv(csv,encoding="utf-8")
print(df)

# check max rows from table
print(df.head())

print(df.head(10))   # first 10 rows

print(df.tail())  # last 5 rows

print(df.tail(10)) # last 10 rows

print(df.info())