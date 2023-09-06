import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import levene, ttest_ind, mannwhitneyu, pearsonr, shapiro

# Cargar el archivo CSV en un DataFrame
df = pd.read_csv("C:/Users/Pablo/Desktop/SZ/hcgalvan_project/data/union/End/dataset_final.csv")

# Definir las variables predictoras
predictors = ['ccbd_iso', 'ccbd_qa']
outcome = 'ccbd_diameter'

# Seleccionar la columna de interés
column_name = "ccbd_diameter"

# Filtrar el DataFrame para obtener los grupos de casos (expuestos) y controles (no expuestos)
grupo_expuesto = df[df['label'] == 1]
grupo_no_expuesto = df[df['label'] == 0]

# regresión lineal para el grupo expuesto
X_expuesto = sm.add_constant(grupo_expuesto[predictors])
y_expuesto = grupo_expuesto[outcome]
model_expuesto = sm.OLS(y_expuesto, X_expuesto).fit()

# regresión lineal para el grupo de controles
X_no_expuesto = sm.add_constant(grupo_no_expuesto[predictors])
y_no_expuesto = grupo_no_expuesto[outcome]
model_no_expuesto = sm.OLS(y_no_expuesto, X_no_expuesto).fit()

# Calcular los valores ajustados (predichos) por cada modelo
predicted_expuesto = model_expuesto.predict(X_expuesto)
predicted_no_expuesto = model_no_expuesto.predict(X_no_expuesto)

# Asegurarse de que ambos grupos tengan la misma cantidad de elementos
min_len = min(len(predicted_expuesto), len(predicted_no_expuesto))
predicted_expuesto = predicted_expuesto[:min_len]
predicted_no_expuesto = predicted_no_expuesto[:min_len]

# Calcular la correlación
correlation, p_value = pearsonr(predicted_expuesto, predicted_no_expuesto)

# Imprimir resumen de los modelos
print("Modelo para Grupo Expuesto:")
print(model_expuesto.summary())
print()

print("Modelo para Grupo No Expuesto:")
print(model_no_expuesto.summary())
print()

print(f"Correlación entre grupos: {correlation}")
print(f"Valor p: {p_value}")

r2_adj_expuesto = model_expuesto.rsquared_adj
r2_adj_no_expuesto = model_no_expuesto.rsquared_adj

if r2_adj_expuesto > r2_adj_no_expuesto:
    print("El modelo para el Grupo Expuesto tiene un mejor ajuste (R-cuadrado ajustado más alto).")
else:
    print("El modelo para el Grupo No Expuesto tiene un mejor ajuste (R-cuadrado ajustado más alto).")
    
    # Estadísticas F y valores p para ambos modelos
f_stat_expuesto = model_expuesto.fvalue
p_value_expuesto = model_expuesto.f_pvalue

f_stat_no_expuesto = model_no_expuesto.fvalue
p_value_no_expuesto = model_no_expuesto.f_pvalue

if p_value_expuesto < p_value_no_expuesto:
    print("El modelo para el Grupo Expuesto es estadísticamente más significativo (valor p más bajo).")
else:
    print("El modelo para el Grupo No Expuesto es estadísticamente más significativo (valor p más bajo).")
    
    # Coeficientes de variables predictoras para ambos modelos
coef_expuesto = model_expuesto.params
coef_no_expuesto = model_no_expuesto.params

# Por ejemplo, si quieres comparar el coeficiente de 'ccbd_iso'
coef_ccbd_iso_expuesto = coef_expuesto['ccbd_iso']
coef_ccbd_iso_no_expuesto = coef_no_expuesto['ccbd_iso']

if abs(coef_ccbd_iso_expuesto) > abs(coef_ccbd_iso_no_expuesto):
    print("El coeficiente de 'ccbd_iso' en el modelo para el Grupo Expuesto es mayor en magnitud.")
else:
    print("El coeficiente de 'ccbd_iso' en el modelo para el Grupo No Expuesto es mayor en magnitud.")
