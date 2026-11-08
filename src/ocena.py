
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Wczytanie danych
iris = load_iris()
X = iris.data
y = iris.target

# Podział danych
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Trenowanie modelu
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Przewidywanie dla danych testowych
y_pred = model.predict(X_test)

# Obliczenie metryk
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
)

# Przygotowanie raportu
wyniki = (
    f"Accuracy: {accuracy:.4f}\n\n"
    f"Classification report:\n{report}\n"
)

# Wyświetlenie wyników
print(wyniki)

# Zapis do pliku
with open("wyniki.txt", "w", encoding="utf-8") as file:
    file.write(wyniki)

print("Wyniki zapisano w pliku wyniki.txt")
