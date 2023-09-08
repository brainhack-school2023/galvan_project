import pandas as pd
import numpy as np
import sys
assert sys.version_info >= (3, 5)
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# Cargar el archivo CSV en un DataFrame
df = pd.read_csv("C:/Users/Pablo/Desktop/SZ/hcgalvan_project/data/union/End/dataset_combinado.csv")

# Especificar las variables predictoras y la variable objetivo
predictors = ['ccbd_qa', 'age', 'ccbd_diameter']  
outcome = 'label'

# Codificación One-Hot para la variable 'gender'
df = pd.get_dummies(df, columns=['gender'])

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(df[predictors + ['gender_F', 'gender_M']], df[outcome], test_size=0.2, random_state=42)

# Lista de penalizaciones a probar
penalties = ['l1', 'l2', 'elasticnet', 'none']

# Iterar a través de las penalizaciones y ajustar modelos de regresión logística
for penalty in penalties:
    print(f"Penalty: {penalty}")
    
    # Crear un modelo de regresión logística con la penalización actual
    if penalty == 'elasticnet':
        model = LogisticRegression(penalty=penalty, solver='saga', l1_ratio=0.5, max_iter=10000)
    else:
        model = LogisticRegression(penalty=penalty, solver='saga', max_iter=10000)

    # Entrenar el modelo
    model.fit(X_train, y_train)

    # Predecir las etiquetas en el conjunto de prueba
    y_pred = model.predict(X_test)

    # Informe de clasificación
    print("Informe de Clasificación:")
    print(classification_report(y_test, y_pred))

    # AUC-ROC
    y_prob = model.predict_proba(X_test)[:, 1]
    auc_roc = roc_auc_score(y_test, y_prob)
    print(f"AUC-ROC: {auc_roc}")
    print("\n")
