import pandas as pd
from sklearn.datasets import load_iris

# Wczytanie zbioru danych
iris = load_iris()

# Utworzenie DataFrame
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# Dodanie kolumny z gatunkiem kwiatu
df["target"] = iris.target

# Eksploracja danych
print("Pierwsze 5 wierszy:")
print(df.head())

print("\nRozmiar zbioru danych:")
print(df.shape)

print("\nStatystyki opisowe:")
print(df.describe())