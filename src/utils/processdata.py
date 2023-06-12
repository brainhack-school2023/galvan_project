import os
import pandas as pd


folder_paths = '../data/datasets'

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
    result1.to_csv('../data/union/End/dataset_final.csv', index=False)
    return result1
