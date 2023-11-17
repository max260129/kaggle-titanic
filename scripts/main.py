import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Charger les données
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# Prétraitement simple
def preprocess_data(data):
    # Sélectionner les colonnes pertinentes
    features = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
    # Convertir les variables catégorielles
    features = pd.get_dummies(features)
    # Gérer les valeurs manquantes
    features['Age'].fillna(features['Age'].median(), inplace=True)
    features['Fare'].fillna(features['Fare'].median(), inplace=True)
    return features

# Prétraiter les données
X = preprocess_data(train_data)
y = train_data['Survived']

# Diviser les données pour la validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer et entraîner le modèle
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Évaluer le modèle
predictions = model.predict(X_val)
accuracy = accuracy_score(y_val, predictions)
print(f"Précision du modèle : {accuracy:.2f}")

# Préparation de la soumission
X_test = preprocess_data(test_data)
test_predictions = model.predict(X_test)
submission = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': test_predictions})

# Enregistrer les prédictions pour la soumission
submission.to_csv('data/submission.csv', index=False)
print("Prédiction sauvegardée dans 'data/submission.csv'")