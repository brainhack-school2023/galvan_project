import pandas as pd

# Cargar DataFrame
df = pd.read_csv("C:/Users/Pablo/Desktop/SZ/hcgalvan_project/data/union/End/dataset_final_con_kmeans.csv")

# Crear una tabla de contingencia entre las columnas 'k_means_group' y 'label'
contingency_table = pd.crosstab(df['k_means_group'], df['label'])

print("Tabla de Contingencia:")
print(contingency_table)

# Extraer las frecuencias para cada combinación
freq_group0_exp = contingency_table.loc[0, 1]
freq_group0_noexp = contingency_table.loc[0, 0]
freq_group1_exp = contingency_table.loc[1, 1]
freq_group1_noexp = contingency_table.loc[1, 0]

# Calcular el Odds Ratio
odds_ratio = (freq_group0_exp * freq_group1_noexp) / (freq_group1_exp * freq_group0_noexp)

print("Odds Ratio:", odds_ratio)