
import os
import pandas as pd

from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Wczytanie konfiguracji z pliku .env
load_dotenv()

data_path = os.getenv("DATA_PATH", "data/iris.csv")
test_size = float(os.getenv("TEST_SIZE", "0.2"))
random_seed = int(os.getenv("RANDOM_SEED", "42"))

# Walidacja konfiguracji
if not 0 < test_size < 1:
    raise ValueError("TEST_SIZE musi być większe od 0 i mniejsze od 1.")

# Wczytanie danych
df = pd.read_csv(data_path)

X = df.drop(columns=["target"])
y = df["target"]

# Podział danych
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=random_seed,
    stratify=y
)

# Trenowanie modelu
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Ocena modelu
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

report = classification_report(
    y_test,
    y_pred,
    target_names=["setosa", "versicolor", "virginica"]
)

wyniki = (
    f"Ścieżka do danych: {data_path}\n"
    f"Test size: {test_size}\n"
    f"Random seed: {random_seed}\n\n"
    f"Accuracy: {accuracy:.4f}\n\n"
    f"Classification report:\n{report}\n"
)

print(wyniki)

with open("wyniki.txt", "w", encoding="utf-8") as file:
    file.write(wyniki)

print("Wyniki zapisano w wyniki.txt")
