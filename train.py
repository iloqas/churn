import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score
import joblib

# --- Fonctions principales ---

def read_data(filepath='data/customer_churn.csv'):
    """
    Lire les données à partir d'un fichier CSV.
    """
    return pd.read_csv(filepath)


def save_model(model, filepath='data/model.pkl'):
    """
    Sauvegarder un modèle entraîné dans un fichier.
    """
    joblib.dump(model, filepath)


def read_model(filepath='data/model.pkl'):
    """
    Charger un modèle sauvegardé.
    """
    return joblib.load(filepath)


def evaluate_model(model, X_test, y_test):
    """
    Évaluer le modèle sur les données de test et afficher les métriques.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print(f"Accuracy du modèle : {accuracy}")
    print(f"Rappel du modèle : {recall}")
    return accuracy, recall


def perform_cross_validation(model, X, y, cv=5):
    """
    Effectuer la validation croisée et afficher les scores.
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"Scores de validation croisée : {scores}")
    print(f"Score moyen : {scores.mean()}")

# --- Script principal ---
if __name__ == "__main__":
    # 1. Lire les données
    df = read_data()

    # 2. Préparation des données
    features = ['Age', 'Total_Purchase', 'Years', 'Num_Sites', 'Account_Manager']
    target = 'Churn'
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Entraînement du modèle
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 4. Évaluation du modèle
    accuracy, recall = evaluate_model(model, X_test, y_test)

    # 5. Validation croisée
    perform_cross_validation(model, X, y)

    # 6. Sauvegarder le modèle
    save_model(model)

    # 7. Charger et tester le modèle sauvegardé
    loaded_model = read_model()
    _, recall_loaded_model = evaluate_model(loaded_model, X_test, y_test)
    print(f"Rappel du modèle chargé : {recall_loaded_model}")
