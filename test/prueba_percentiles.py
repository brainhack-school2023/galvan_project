import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import ttest_ind, mannwhitneyu
from scipy.stats import levene
from scipy.stats import shapiro
import sys
assert sys.version_info >= (3, 5)
import os

# Cargar el archivo CSV en un DataFrame Corregir ruta
df = pd.read_csv("C:/Users/Pablo/Desktop/SZ/hcgalvan_project/data/union/End/dataset_final.csv")

num_rows = 49  # Cambia esto al número deseado de filas

# Tomar un subconjunto de filas del DataFrame
subset_data = df.head(num_rows)

# Seleccionar la columna de interés para el cálculo de percentiles
column_name = "ccbd_diameter" 

data = subset_data[column_name].values
p3 = np.percentile(data, 3)
print("percentil 3 (P3):", p3)

# Calcular el percentil 25 (primer cuartil)
q1 = np.percentile(data, 25)
print("Primer cuartil (Q1):", q1)

# Calcular el percentil 50 (mediana)
median = np.percentile(data, 50)
print("Mediana:", median)

# Calcular el percentil 75 (tercer cuartil)
q3 = np.percentile(data, 75)
print("Tercer cuartil (Q3):", q3)

# Calcular el percentil 97 (P97)
p97 = np.percentile(data, 97)
print("Percentil 97 (P97):", p97)

# Calcular el percentil 95 (P95)
p95 = np.percentile(data, 95)
print("Percentil 95 (P95):", p95)

# Calcular el percentil 5 (P5)
p5 = np.percentile(data, 5)
print("Percentil 5 (P5):", p5)


#Lo siguiente tiene un error de concepto en expuesto y no expuesto
columna_exposicion = "ccbd_diameter"
columna_etiquetas = "label"

# Filtrar el DataFrame para obtener los grupos expuesto y no expuesto
grupo_expuesto = df[df[columna_etiquetas] == 1]
grupo_no_expuesto = df[df[columna_etiquetas] == 0]

# Extraer las medidas de exposición para ambos grupos
exposicion_expuesto = grupo_expuesto[columna_exposicion]
exposicion_no_expuesto = grupo_no_expuesto[columna_exposicion]

# estadísticas descriptivas para ambos grupos
media_expuesto = exposicion_expuesto.mean()
mediana_expuesto = exposicion_expuesto.median()
media_no_expuesto = exposicion_no_expuesto.mean()
mediana_no_expuesto = exposicion_no_expuesto.median()

print("Grupo Expuesto:")
print("Media de la exposición:", media_expuesto)
print("Mediana de la exposición:", mediana_expuesto)
print()

print("Grupo No Expuesto:")
print("Media de la exposición:", media_no_expuesto)
print("Mediana de la exposición:", mediana_no_expuesto)


#Distribución de la Exposición en Grupos
sns.histplot(data=grupo_expuesto, x=column_name, kde=True, label='Grupo Expuesto')
sns.histplot(data=grupo_no_expuesto, x=column_name, kde=True, label='Grupo No Expuesto')
plt.xlabel('Exposición')
plt.ylabel('Densidad')
plt.title('Distribución de la Exposición en Grupos')
plt.legend()
plt.show()

#Shapiro
stat_expuesto, p_expuesto = shapiro(grupo_expuesto[column_name])
stat_no_expuesto, p_no_expuesto = shapiro(grupo_no_expuesto[column_name])

print(f'Grupo Expuesto - p-valor Shapiro-Wilk: {p_expuesto}')
print(f'Grupo No Expuesto - p-valor Shapiro-Wilk: {p_no_expuesto}')


#p Levene
stat_levene, p_levene = levene(grupo_expuesto[column_name], grupo_no_expuesto[column_name])

print(f'Valor p de Levene: {p_levene}')

#significancia normalidad
if p_expuesto > 0.05 and p_no_expuesto > 0.05 and p_levene > 0.05:
    # Realizar prueba t de Student
    t_stat, p_ttest = ttest_ind(grupo_expuesto[column_name], grupo_no_expuesto[column_name])
    test_used = 't de Student'
else:
    # Realizar prueba de Mann-Whitney U
    u_stat, p_mannwhitneyu = mannwhitneyu(grupo_expuesto[column_name], grupo_no_expuesto[column_name], alternative='two-sided')
    test_used = 'Mann-Whitney U'

if test_used == 't de Student':
    print(f'Prueba t de Student:')
    print(f'Estadístico t: {t_stat}')
    print(f'Valor p: {p_ttest}')
else:
    print(f'Prueba de Mann-Whitney U:')
    print(f'Estadístico U: {u_stat}')
    print(f'Valor p: {p_mannwhitneyu}')
    
    transformed_data = np.sqrt(data)  # Por ejemplo, transformación raíz cuadrada
statistic, p_value = stats.shapiro(transformed_data)


X_expuesto = sm.add_constant(grupo_expuesto[['ccbd_iso', 'ccbd_qa']])
X_no_expuesto = sm.add_constant(grupo_no_expuesto[['ccbd_iso', 'ccbd_qa']])

# Ajustar modelos de regresión lineal
model_expuesto = sm.OLS(exposicion_expuesto, X_expuesto).fit()
model_no_expuesto = sm.OLS(exposicion_no_expuesto, X_no_expuesto).fit()

# Imprimir resumen de los modelos
print("Modelo para Grupo Expuesto:")
print(model_expuesto.summary())
print()

print("Modelo para Grupo No Expuesto:")
print(model_no_expuesto.summary())
print()
# fin errores - En proceso de corrección

# Definir las variables predictoras y la variable de interés
predictors = ['ccbd_iso', 'ccbd_qa']
outcome = 'ccbd_diameter'

# Filtrar el DataFrame para obtener los grupos de casos (expuestos) y controles (no expuestos)
grupo_expuesto = df[df['label'] == 'casos']
grupo_no_expuesto = df[df['label'] == 'controles']

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

# Calcular la correlación entre los valores ajustados de ambos grupos
correlation, p_value = pearsonr(predicted_expuesto, predicted_no_expuesto)

print(f"Correlación entre grupos: {correlation}")
print(f"Valor p: {p_value}")