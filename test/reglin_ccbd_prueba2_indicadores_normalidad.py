import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import levene, pearsonr, shapiro

# Cargar el archivo CSV en un DataFrame
df = pd.read_csv("C:/hcgalvan/repositorios/hcgalvan_project/data/union/End/dataset_final.csv")

# Definir las variables predictoras
predictors = ['ccbd_iso', 'ccbd_qa']
outcome = 'ccbd_diameter'

# Filtrar el DataFrame para obtener los grupos de casos (expuestos) y controles (no expuestos)
grupo_expuesto = df[df['label'] == 1]
grupo_no_expuesto = df[df['label'] == 0]

# Regresión lineal para el grupo expuesto
X_expuesto = sm.add_constant(grupo_expuesto[predictors])
y_expuesto = grupo_expuesto[outcome]
model_expuesto = sm.OLS(y_expuesto, X_expuesto).fit()

# Regresión lineal para el grupo de controles
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

# Calcular la correlación entre los valores ajustados de ambos grupos
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

# Prueba de normalidad de los residuos (Shapiro-Wilk)
residuos_expuesto = model_expuesto.resid
shapiro_test_stat, shapiro_p_value = shapiro(residuos_expuesto)

if shapiro_p_value > 0.05:
    print("Los residuos siguen una distribución normal (p > 0.05)")
else:
    print("Los residuos no siguen una distribución normal (p <= 0.05)")

# Prueba de homocedasticidad (Levene)
homocedasticidad_test_stat, homocedasticidad_p_value = levene(residuos_expuesto, predicted_no_expuesto)

if homocedasticidad_p_value > 0.05:
    print("Homocedasticidad se cumple (p > 0.05)")
else:
    print("Homocedasticidad no se cumple (p <= 0.05)")

# Asegurarse de que ambos grupos tengan la misma cantidad de elementos
min_len = min(len(predicted_expuesto), len(residuos_expuesto))
predicted_expuesto = predicted_expuesto[:min_len]
residuos_expuesto = residuos_expuesto[:min_len]

# Gráfico de dispersión de residuos vs. valores predichos
plt.scatter(predicted_expuesto, residuos_expuesto)
plt.xlabel('Valores Predichos')
plt.ylabel('Residuos')
plt.title('Gráfico de Residuos vs. Valores Predichos')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()

# Gráfico Q-Q de los residuos
import statsmodels.api as sm
sm.qqplot(residuos_expuesto, line='s')
plt.title('Gráfico Q-Q de Residuos')
plt.show()
