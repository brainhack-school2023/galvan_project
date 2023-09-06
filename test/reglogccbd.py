import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import levene, pearsonr, shapiro
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Cargar el archivo CSV en un DataFrame
df = pd.read_csv("C:/Users/Pablo/Desktop/SZ/hcgalvan_project/data/union/End/dataset_final.csv")

# Especificar los cuantiles como puntos de corte
cuantiles = [0, 0.1, 0.25, 0.5, 0.75, 0.95, 1.0]  # Puedes ajustar estos valores según tus necesidades

# Calcular los cuantiles
cut_points = df['ccbd.dm'].quantile(cuantiles)

# Agregar etiquetas para los intervalos
labels = [f'Q{i}' for i in range(1, len(cut_points))]

# Discretizar la variable continua en intervalos
df['ccbd_dm_interval'] = pd.cut(df['ccbd.dm'], bins=cut_points, labels=labels, include_lowest=True)

# Imprimir el DataFrame resultante
print(df[['ccbd.dm', 'ccbd_dm_interval']])

# Definir las variables predictoras
predictors = ['ccbd_iso', 'ccbd_qa', 'ccbd_diameter_bin_11-20', 'ccbd_diameter_bin_21-30', 'ccbd_diameter_bin_31-40', 'ccbd_diameter_bin_41-50']
outcome = 'label'  # Cambia la variable dependiente a 'label', que es binaria (0 o 1)

# Regresión logística para predecir 'label' en función de los predictores
X = df[predictors]
y = df[outcome]
model = LogisticRegression()
model.fit(X, y)

# Calcular las probabilidades ajustadas (predichas)
predicted_prob = model.predict_proba(X)[:, 1]

# Imprimir resumen del modelo
print("Modelo Logístico:")
print(classification_report(y, model.predict(X)))
print()

# Gráfico de dispersión de probabilidades ajustadas vs. valores predichos
plt.scatter(predicted_prob, y)
plt.xlabel('Probabilidades Ajustadas')
plt.ylabel('Etiquetas Reales')
plt.title('Gráfico de Probabilidades Ajustadas vs. Etiquetas Reales')
plt.show()
