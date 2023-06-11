from sklearn.impute import KNNImputer

def missing_values(train_set):
    # Reemplazamos los valores faltantes por np.nan
    # Construimos el modelo
    imputer = KNNImputer(n_neighbors=5, weights="uniform")
    # Ajustamos el modelo e imputamos los missing values
    for i in train_set.columns:
        if train_set[i].isnull().sum() > 0:
            imputer.fit(train_set[[i]])
            train_set[i] = imputer.transform(train_set[[i]]).ravel()
    return train_set