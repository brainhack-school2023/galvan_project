import os
import pandas as pd
import numpy as np
from utils import labelpre as lp
from utils import clean as cl
import seaborn as sns

folder_paths = '../data/datasets'
folder_paths_cobre = '../data/cobre'

def read_txt_to_df(file_path):
    with open(file_path, 'r') as f:
        data = f.read()
    df = pd.DataFrame([data.split('\n')])
    return df

def transpose_df(df):
    return df.transpose()

# hacer listado de subcarpetas
def dataread():
    df_list = []
    for root, dirs, files in os.walk(folder_paths):
        for file_name in files:
            if file_name.endswith('.txt'):
                file_path = os.path.join(root, file_name)
                df = read_txt_to_df(file_path)
                
                df = transpose_df(df)
                df['N'] = file_name+"\t"+df.index.astype(str)
                df_list.append(df)
    result = pd.concat(df_list)
    result1 = result.apply(lambda x: pd.Series(x.dropna().values))
    result1.rename(columns = {0:'data', 'N':'label'}, inplace = True)
    result1[['Codigo','values']]= result1.data.str.split(pat='\t',expand=True)
    result1.dropna(inplace=True)
    result1[['Et','id']]= result1.label.str.split(pat='\t',expand=True)
    #result1.to_csv('../data/union/End/dataset_final.csv', index=False)
    return result1

### DATOS EXTRAIDOS DE UCLA
def datast(data):
# Cambia los nombres, elimina columnas y transpone    
    conditionlist = [
            (data['Et'].str.contains('Arcuate_Fasciculus_L')),
            (data['Et'].str.contains('Arcuate_Fasciculus_R')),
            (data['Et'].str.contains('Cingulum_Frontal_Parietal_L')),
            (data['Et'].str.contains('Cingulum_Frontal_Parietal_R')),
            (data['Et'].str.contains('Corpus_Callosum_Body')),#Nuevo
            (data['Et'].str.contains('Extreme_Capsule_L')),#Nuevo
            (data['Et'].str.contains('Frontal_Aslant_Tract_L')),
            (data['Et'].str.contains('Frontal_Aslant_Tract_R')),
            (data['Et'].str.contains('Inferior_Fronto_Occipital_Fasciculus_L')),#Nuevo
            (data['Et'].str.contains('Superior_Longitudinal_Fasciculus1_L')),
            (data['Et'].str.contains('Superior_Longitudinal_Fasciculus1_R')),
            (data['Et'].str.contains('Superior_Longitudinal_Fasciculus2_L')), #Nuevo
            (data['Et'].str.contains('Thalamic_Radiation_Anterior_L')), #Nuevo
            (data['Et'].str.contains('Uncinate_Fasciculus_L')),
            (data['Et'].str.contains('Uncinate_Fasciculus_R')) ]

    choicelist = ['afsl_',
            'afsr_',
            'cfpl_',
            'cfpr_',
            'ccbd_', #Nuevo
            'ecpl_', #Nuevo
            'fatl_',
            'fatr_',
            'ifol_', #Nuevo
            'slfl_',
            'slfr_',
            'sl2l_', #Nuevo
            'tral_', #Nuevo
            'ufsl_',
            'ufsr_']
    data.reindex(columns=['Et'])
    data['Nc'] = np.select(conditionlist, choicelist, default='Not Specified')
    data['Ncodigo'] = data['Nc']+data['Codigo']
    data[['cod','scrup']]= data['Et'].str.split(r'[\_]dwi', expand=True)
    data.drop(columns = ['data','label','Codigo','Et','Nc','scrup'], inplace=True)
    data.reindex(columns=['cod'])
    # pivotamos la tabla subject para que los valores de la columna Ncodigo se conviertan en columnas
    datf = data.pivot(index=['cod'], columns='Ncodigo', values='values').reset_index()
    columnas = lp.custom()
    datf.rename(columns=columnas, inplace=True)
    datf['label'] = datf.cod.str.contains('sub-10').astype(int)
    sns.heatmap(datf.isnull(), cbar=False)
    datf = cl.missing_values(datf)
    datf.to_csv('../data/union/End/dataset_final.csv', index=False)
    
    return datf
### DATOS EXTRAIDOS DE COBRE
# hacer listado de subcarpetas
def dataread_cobre():
    df_list = []
    for root, dirs, files in os.walk(folder_paths_cobre):
        for file_name in files:
            if file_name.endswith('.txt'):
                file_path = os.path.join(root, file_name)
                df = read_txt_to_df(file_path)
                
                df = transpose_df(df)
                df['N'] = file_name+"\t"+df.index.astype(str)
                df_list.append(df)
    result = pd.concat(df_list)
    result1 = result.apply(lambda x: pd.Series(x.dropna().values))
    result1.rename(columns = {0:'data', 'N':'label'}, inplace = True)
    result1[['Codigo','values']]= result1.data.str.split(pat='\t',expand=True)
    result1.dropna(inplace=True)
    result1[['Et','id']]= result1.label.str.split(pat='\t',expand=True)
    #result1.to_csv('../data/union/End/dataset_final.csv', index=False)
    return result1

def datast_cobre(data):
# Cambia los nombres, elimina columnas y transpone    
    conditionlist = [
            (data['Et'].str.contains('Corpus_Callosum_Body')),#Nuevo
            (data['Et'].str.contains('Extreme_Capsule_L')),#Nuevo
            (data['Et'].str.contains('Frontal_Aslant_Tract_L')),
            (data['Et'].str.contains('Inferior_Fronto_Occipital_Fasciculus_L')),#Nuevo
            (data['Et'].str.contains('Superior_Longitudinal_Fasciculus2_L')), #Nuevo
            (data['Et'].str.contains('Thalamic_Radiation_Anterior_L')) ]
    
    choicelist = ['ccbd_', #Nuevo
            'ecpl_', #Nuevo
            'fatl_',
            'ifol_', #Nuevo
            'sl2l_', #Nuevo
            'tral_']
    
    data.reindex(columns=['Et'])
    data['Nc'] = np.select(conditionlist, choicelist, default='Not Specified')
    data['Ncodigo'] = data['Nc']+data['Codigo']
    data[['cod','scrup']]= data['Et'].str.split(r'[\_]dwi', expand=True)
    data.drop(columns = ['data','label','Codigo','Et','Nc','scrup'], inplace=True)
    data.reindex(columns=['cod'])
    # pivotamos la tabla subject para que los valores de la columna Ncodigo se conviertan en columnas
    datf = data.pivot(index=['cod'], columns='Ncodigo', values='values').reset_index()
    columnas = lp.custom()
    datf.rename(columns=columnas, inplace=True)
    datf['label'] = datf.cod.str.contains('sub-A0000').astype(int)
    sns.heatmap(datf.isnull(), cbar=False)
    datf = cl.missing_values(datf)
    datf.to_csv('../data/union/End/dataset_final_cobre.csv', index=False)
    
    return datf