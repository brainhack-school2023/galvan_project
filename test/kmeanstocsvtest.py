import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 
import sys
assert sys.version_info >= (3, 5)
import os

# Cargar el archivo CSV en un DataFrame
df = pd.read_csv("C:/Users/Pablo/Desktop/SZ/hcgalvan_project/data/union/End/dataset_final.csv")

# Seleccionar la columna de interés
column_name = "ccbd_diameter"

num_rows = 49  # Cambia esto al número deseado de filas

# Tomar un subconjunto de filas del DataFrame
subset_data = df.head(num_rows)


# Preparar los datos para el clustering
X = df[[column_name]].values

# Determinar el número de clusters (grupos) deseado
num_clusters = 2

# Aplicar el algoritmo K-Means
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['grupo'] = kmeans.fit_predict(X)

# Agregar la columna "k_means_group" al DataFrame
df['k_means_group'] = kmeans.labels_

# Mostrar información sobre los grupos
group_summary = df.groupby('grupo')[column_name].describe()
print(group_summary)

# Visualizar los grupos en un diagrama de dispersión
plt.scatter(df.index, df[column_name], c=df['grupo'], cmap='viridis')
plt.xlabel('Muestra')
plt.ylabel(column_name)
plt.title('Diagrama de Dispersión de Grupos')
plt.show()

# Guardar el DataFrame modificado en el archivo CSV
df.to_csv("C:/Users/Pablo/Desktop/SZ/hcgalvan_project/data/union/End/dataset_final_con_kmeans.csv", index=False)
