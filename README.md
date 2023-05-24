
## Schizophrenia prediction: use of Neuroimages and Artificial Intelligence Models.
~~Detection of schizophrenia based on tractographic techniques with fine tuning in machine learning.~~


# 

| Title        | repo    | tags | date
|:--------------------|:--------------------:|:--------------------:|:--------------------
|Pitch Project  |[**Open In Google**](https://docs.google.com/presentation/d/1pgS_7BF-ZuwxwgHhbA5jdm0gcpsiapIzR7eVAEqdzcQ/edit?usp=sharing) | Neuroimages | 2023-05-24
| Dataset | [**Open In Google**](https://purl.stanford.edu/mg599hw5271)| UCLA | 2023-05-24



## Project definition
#### Main Question
Is it possible to combine tractography analysis, fMRI and Machine Learning to create an artificial intelligence model in order to enhance schizophrenia prediction?



### Background
**Schizophrenia theories**: Theory of Disconnection Syndrome
* The theory of disconnection syndrome is a theory that explains the symptoms of schizophrenia as the result of disruptions in the normal integration of emotion, perception, and thought.

**Previous works and literature**: Use of only one method of analyses, Require additional validation afterwards
* [Schizophrenia prediction using tractography and machine learning](https://www.sciencedirect.com/science/article/pii/S1053811919303837)
* [Schizophrenia prediction using fMRI and machine learning](https://www.sciencedirect.com/science/article/pii/S1053811919303837)

**MRI**: Now it’s only use for differential diagnosis and not for prediction


### Tools


### Methodology
* Preprocessing, processing and tractography and fMRI
using software DSI Studio and SPM 
* Creating and training AI model
* Statistical analyses
### Data
* [UCLA Consortium for Neuropsychiatric Phenomics LA5c Study](https://purl.stanford.edu/mg599hw5271)
### Objectives
**In Spanish:**
* Diseñar un modelo de aprendizaje automático que detecte anomalías y mecanismos subyacentes a traves de la conectómica de la corteza prefrontal, luego clasifique en categorías, y finalmente prediga el diagnóstico de la enfermedad de esquizofrenia.

Específicamente se propone:
1. Utilizar técnicas de tractografía en resonancias magnéticas  estructurales y por tensor de difusión para capturar la conectividad estructural y transformar en dataset para aprendizaje automatico.
* Evaluar las conexiones existentes del PFC:
   * Fascículo arqueado
   * Parietal frontal del cíngulo
   * SLF
   * Fascículo uncinado
   * Red de modo predeterminado
2. Crear un script que separa los datos en entrenamiento y prueba, y que se pueda modificar el tamaño de la muestra.
3. Comparar modelos generales de clasificación para determinar el mejor modelo.
4. Comparar El conectoma de sujeto sano (grupo 1) con sujetos con diagnóstico de esquizofrenia (grupo 2)

**An English:**
* Design a machine learning model that detects abnormalities and underlying mechanisms through connectomics of the prefrontal cortex, then classifies them into categories, and finally predicts the diagnosis of schizophrenia illness.

Specifically:
1. Use tractography techniques in structural and diffusion tensor MRI to capture structural connectivity, also, and transform into dataset for machine learning.
* Evaluate he existing connections of the PFC:
   * Arcuate Fasciculus
   * Cingulum Frontal Parietal
   * SLF
   * Uncinate Fasciculus
   * Default mode network
2. Create a script that separates the data in training and test, and that can modify the size of the sample.
3. Compare general classification models to determine the best model.
4. Compare The connectome of healthy subject (group 1) with subjects with a diagnosis of schizophrenia (group 2)


### Deliverables
* A Github repository with codes and scripts to reproduce training and testing.
* A jupyter notebook of the analysis codes and visualisations for comparing the results.
* Documentation

### Colaborators

|H.Galván|A.Boveda|P.Koss|S.Galván|
|:--------------------|:--------------------:|:--------------------:|:--------------------
|<a href="https://github.com/hcgalvan"><img src="https://avatars.githubusercontent.com/hcgalvan" width="80px;" alt=""/>|<a href="https://github.com/agustinabl"><img src="https://avatars.githubusercontent.com/agustinabl" width="80px;" alt=""/>|<a href="https://github.com/pablokoss"><img src="https://avatars.githubusercontent.com/pablokoss" width="80px;" alt=""/>|<a href="https://github.com/dseba9"><img src="https://avatars.githubusercontent.com/dseba9" width="80px;" alt=""/>|
   