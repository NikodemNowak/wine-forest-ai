import pandas as pd
import seaborn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix

# === Wczytanie i przygotowanie danych ===

try:
    sciezka_pliku = 'winequality-white.csv'
    dane = pd.read_csv(sciezka_pliku, sep=';')
    print(f"Dane zostały pomyślnie wczytane. Liczba próbek: {len(dane)}")
except FileNotFoundError:
    print(f"BŁĄD: Nie znaleziono pliku '{sciezka_pliku}'.")
    print("Upewnij się, że plik CSV znajduje się w tym samym folderze co skrypt Pythona.")
    exit()

# Przygotowanie danych do modelu:
# 'X' - cechy (dane wejściowe), czyli wszystkie kolumny oprócz 'quality'.
# .drop() do usunięcia kolumny z naszym celem.
X = dane.drop('quality', axis=1)

# 2. 'y' - etykiety, czyli tylko kolumna 'quality'.
y = dane['quality']

# Podział danych na zbiór treningowy i testowy.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Zbiór danych podzielony na {len(X_train)} próbek treningowych i {len(X_test)} próbek testowych.\n")

# === Trenowanie i ocena modelu - Drzewo Decyzyjne ===
print("------------------------------")
print("Trenowanie modelu: Drzewo Decyzyjne...")

# Tworzymy obiekt modelu Drzewa Decyzyjnego.
drzewo_decyzyjne = DecisionTreeClassifier(random_state=42, min_samples_leaf=3)

# Trenujemy model na danych treningowych
drzewo_decyzyjne.fit(X_train, y_train)

# Używamy wytrenowanego modelu
prognozy_drzewo = drzewo_decyzyjne.predict(X_test)

# Obliczamy dokładność (accuracy), czyli jaki procent przewidywań był poprawny.
dokladnosc_drzewo = accuracy_score(y_test, prognozy_drzewo)

# Wypisanie wszystko unikalnych wartości quality z danych
print("Unikalne wartości w kolumnie 'quality':", sorted(y.unique()))
print(f"Zbiór treningowy: {len(X_train)} próbek, klasy: {sorted(y_train.unique())}")
print(f"Zbiór testowy:     {len(X_test)} próbek, klasy: {sorted(y_test.unique())}\n")

# Wyświetlamy wynik w procentach.
print(f"Dokładność Drzewa Decyzyjnego: {dokladnosc_drzewo * 100:.2f}%")
print("------------------------------")

# === Trenowanie i ocena modelu - Las Losowy ===
print("Trenowanie modelu: Las Losowy...")

# Tworzymy obiekt modelu Lasu Losowego.
# n_estimators=100 -> model będzie się składał ze 100 drzew decyzyjnych.
las_losowy = RandomForestClassifier(n_estimators=200, random_state=42, min_samples_leaf=2)

# Trenujemy model na tych samych danych treningowych.
las_losowy.fit(X_train, y_train)

# Przewidujemy jakość win na danych testowych za pomocą Lasu Losowego.
prognozy_las = las_losowy.predict(X_test)

# Obliczamy dokładność dla Lasu Losowego.
dokladnosc_las = accuracy_score(y_test, prognozy_las)

# Wyświetlamy wynik w procentach.
print(f"Dokładność Lasu Losowego: {dokladnosc_las * 100:.2f}%")
print("------------------------------\n")

# === Podsumowanie i porównanie wyników ===
print("Podsumowanie:")
print(f"Dokładność Drzewa Decyzyjnego: {dokladnosc_drzewo * 100:.2f}%")
print(f"Dokładność Lasu Losowego:      {dokladnosc_las * 100:.2f}%")


# 1. Małe drzewo do wizualizacji
drzewo_pokazowe = DecisionTreeClassifier(max_depth=3, random_state=42)
drzewo_pokazowe.fit(X_train, y_train)

plt.figure(figsize=(18, 10))
plot_tree(
    drzewo_pokazowe,
    max_depth=3,
    feature_names=X.columns,
    class_names=[f"Quality {c}" for c in sorted(y.unique())],
    filled=True,
    rounded=True,
    fontsize=8,
    precision=1,
    impurity=False,
    label='none'
)
plt.title("Małe Drzewo Decyzyjne (max_depth=3)")
plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.05)
plt.show()

# 2. Ważność cech - Drzewo Decyzyjne
tree_importances = pd.Series(drzewo_decyzyjne.feature_importances_, index=X.columns)
plt.figure(figsize=(10, 6))
tree_importances.sort_values().plot(kind='barh', color='orange')
plt.title("Ważność cech - Drzewo Decyzyjne")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# 3. Macierz pomyłek - Drzewo Decyzyjne
cm_tree = confusion_matrix(y_test, prognozy_drzewo)
plt.figure(figsize=(8, 6))
seaborn.heatmap(
    cm_tree, annot=True, fmt='d', cmap='Oranges',
    xticklabels=sorted(y_test.unique()), yticklabels=sorted(y_test.unique())
)
plt.title("Macierz pomyłek - Drzewo Decyzyjne")
plt.xlabel("Przewidziana klasa")
plt.ylabel("Rzeczywista klasa")
plt.show()

# 4. Ważność cech - Las Losowy
forest_importances = pd.Series(las_losowy.feature_importances_, index=X.columns)
plt.figure(figsize=(10, 6))
forest_importances.sort_values().plot(kind='barh', color='green')
plt.title("Ważność cech - Las Losowy")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# 5. Macierz pomyłek - Las Losowy
cm_forest = confusion_matrix(y_test, prognozy_las)
plt.figure(figsize=(8, 6))
seaborn.heatmap(
    cm_forest, annot=True, fmt='d', cmap='Blues',
    xticklabels=sorted(y_test.unique()), yticklabels=sorted(y_test.unique())
)
plt.title("Macierz pomyłek - Las Losowy")
plt.xlabel("Przewidziana klasa")
plt.ylabel("Rzeczywista klasa")
plt.show()