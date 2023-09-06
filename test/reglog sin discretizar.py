import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import seaborn as sns

# Cargar el archivo CSV en un DataFrame
df = pd.read_csv("C:/Users/Pablo/Desktop/SZ/hcgalvan_project/data/union/End/dataset_final.csv")

# Especificar las variables predictoras y la variable objetivo
predictors = ['ccbd_iso', 'ccbd_diameter']  # Ajusta las variables predictoras
outcome = 'label'

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(df[predictors], df[outcome], test_size=0.2, random_state=42)

# Crear y ajustar el modelo de regresión logística
model = LogisticRegression()
model.fit(X_train, y_train)

# Predecir las etiquetas y las probabilidades en el conjunto de prueba
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

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
