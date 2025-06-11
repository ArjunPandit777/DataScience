import pandas as pd

# Reading JSON file
url = "https://www.thecocktaildb.com/api/json/v1/1/random.php"
data = pd.read_json(url)
print(data.to_string())