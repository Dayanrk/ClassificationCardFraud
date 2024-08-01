import pandas as pd
import numpy as np
import os , sys
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

path = 'creditcard.csv'

df = pd.read_csv(path)
df.isnull().sum()
df = df.dropna()  # Supprime les lignes avec des valeurs manquantes

# Number classes
classes = df['Class'].value_counts()

X = df.iloc[:,:-1]
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

clf = RandomForestClassifier()

param_grid = {
    'max_depth': [1, 2, 6, 10],
    'criterion': ['gini','entropy'],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 4],

}
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1,
                           scoring = {
                               'accuracy': 'accuracy',
                               'precision': 'precision_macro',
                               'recall': 'recall_macro'},
                           refit='precision')
print("Model")
grid_search.fit(X_train, y_train)

# Évaluer le modèle avec les meilleurs paramètres sur l'ensemble de test
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print(grid_search.scorer_)
print("Meilleur modèle : ", best_model)
print("Score sur l'ensemble de test : ", test_score)
