import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import seaborn as sns

# Cargar el archivo CSV en un DataFrame
df = pd.read_csv("C:/Users/Pablo/Desktop/SZ/hcgalvan_project/data/union/End/dataset_final.csv")

# Especificar las variables predictoras y la variable objetivo
predictors = ['ccbd_iso', 'ccbd_qa', 'ccbd_diameter']  # Ajusta las variables predictoras
outcome = 'label'

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(df[predictors], df[outcome], test_size=0.2, random_state=42)

# Definir una función para discretizar basada en frecuencia
def frequency_discretization(data, num_bins):
    # Usar la función cut de pandas para discretizar basado en la frecuencia
    bins = pd.cut(data, bins=num_bins, labels=False)
    return bins

# Especificar el número de bins deseado
num_bins = 6  # Puedes ajustar este valor

# Aplicar la discretización basada en frecuencia a las características
X_train_discretized = X_train.apply(lambda x: frequency_discretization(x, num_bins))
X_test_discretized = X_test.apply(lambda x: frequency_discretization(x, num_bins))

# Entrenar un nuevo modelo de regresión logística con las características discretizadas
model = LogisticRegression()
model.fit(X_train_discretized, y_train)

# Predecir las etiquetas en el conjunto de prueba
y_pred = model.predict(X_test_discretized)

# Calcular métricas y mostrar el informe de clasificación
print("Informe de Clasificación:")
print(classification_report(y_test, y_pred))

# Predecir las probabilidades en el conjunto de prueba
y_prob = model.predict_proba(X_test_discretized)[:, 1]

# Calcular el AUC-ROC
auc_roc = roc_auc_score(y_test, y_prob)
print(f"AUC-ROC: {auc_roc}")


# Calcular métricas y mostrar el informe de clasificación
print("Informe de Clasificación:")
print(classification_report(y_test, y_pred))

# Calcular el AUC-ROC
auc_roc = roc_auc_score(y_test, y_prob)
print(f"AUC-ROC: {auc_roc}")

# Gráfico de dispersión de probabilidades ajustadas vs. etiquetas reales
sns.scatterplot(x=y_prob, y=y_test)
plt.xlabel('Probabilidades Ajustadas')
plt.ylabel('Etiquetas Reales')
plt.title('Gráfico de Probabilidades Ajustadas vs. Etiquetas Reales')
plt.show()


