# Tiene el proceso de entrenamiento
def modelos():
    st.sidebar.title("Visualizar Seteos")
    eleccion = st.sidebar.radio(
                "Modelos",
                ("RidgeClassifier","RandomForestClassifier","GradientBoostingClassifier")
            )
    st.write(eleccion)    

    df = data
    X = df.drop('class', axis=1) # features
    y = df['class'] # target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
    
    pipelines = {
    #'lr':make_pipeline(StandardScaler(), LogisticRegression()),
    'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
    }
    
    fit_models = {}
    for algo, pipeline in pipelines.items():
        model = pipeline.fit(X_train, y_train)
        fit_models[algo] = model
    
    logger.debug("=== Alive threads ===")
    for thread in threading.enumerate():
        if thread.is_alive():
            logger.debug(f"  {thread.name} ({thread.ident})")

    rcp = fit_models['rc'].predict(X_test)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('_X test_')
        st.table(X_test.iloc[0:5])

    with col2:
        st.markdown('_y test_')
        st.table(y_test.iloc[0:5])

    with col3:
        st.markdown('_Predicción_')
        st.table(rcp[0:5])
        
    for algo, model in fit_models.items():
        yhat = model.predict(X_test)
        st.write('Modelo: ',algo, 'Accuracy Score: ', accuracy_score(y_test, yhat),'Precision Score: ',
          precision_score(y_test.values, yhat,average="weighted",pos_label="derecha"),'Recall Score: ',
          recall_score(y_test.values, yhat, average="weighted", pos_label="derecha"))
 