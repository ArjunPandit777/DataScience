import pandas as pd


# table is known as dataframe

employees = {
  "name": ["John", "Jane", "Jack", "Jill"],
  "age": [20, 21, 22, 23],
  "salary": [1000, 2000, 3000, 4000],
  "gender": ["Male", "Female", "Male", "Female"],
  "department": ["IT", "Finance", "HR", "Marketing"],
  "designation": ["Manager", "Analyst", "Manager", "Assistant"],
  "hire_date": ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"],
  "is_active": [True, True, False, True],
  "is_manager": [True, False, True, False],
  "is_senior": [True, False, False, True],
  "is_female": [False, True, False, True]
}

employees = pd.DataFrame(employees)
print(employees)

# get data of single row usinf loc
print(employees.loc[0])

# get multiple rows using loc
print(employees.loc[[0,2]])