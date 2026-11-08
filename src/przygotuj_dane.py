
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["target"] = iris.target

df.to_csv("data/iris.csv", index=False)

print("Dane zapisano w data/iris.csv")
print("Rozmiar zbioru:", df.shape)
