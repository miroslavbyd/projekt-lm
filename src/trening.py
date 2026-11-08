from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Wczytanie danych
iris = load_iris()
X = iris.data
y = iris.target

# Podział na dane treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Utworzenie i trenowanie modelu
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

print("Liczba próbek treningowych:", len(X_train))
print("Liczba próbek testowych:", len(X_test))
print("Model został wytrenowany.")