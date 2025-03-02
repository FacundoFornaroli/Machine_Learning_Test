## IMDB_ML_Prediction.ipynb

Este Jupyter Notebook está diseñado para predecir los sentimientos (positivos o negativos) de las reseñas de películas en IMDB utilizando técnicas de Machine Learning. A continuación, se describen las principales secciones y pasos incluidos en el notebook:

1. **Importar Librerías**: Se importan las librerías necesarias como pandas, imblearn y sklearn para la manipulación de datos y la implementación de modelos de Machine Learning.

2. **Carga y Preparación de Datos**:
   - Se carga el dataset "IMDB Dataset.csv" que contiene 50,000 filas con dos columnas: `review` y `sentiment`.
   - Se desbalancea el dataset seleccionando 9,000 reseñas positivas y 1,000 negativas.

3. **Balanceo de Datos**: 
   - Se utiliza la librería `imblearn` para balancear el dataset mediante undersampling, asegurando que ambas clases (positivas y negativas) tengan la misma cantidad de datos.

4. **Separación de Datos**: 
   - Se separa el dataset balanceado en conjuntos de entrenamiento y prueba.

5. **Representación de Texto**: 
   - Se transforma el texto de las reseñas en datos numéricos utilizando `TfidfVectorizer`.

6. **Modelos de Machine Learning**: 
   - Se implementan y entrenan varios modelos de clasificación supervisada: 
     - Support Vector Machines (SVM)
     - Árbol de Decisión
     - Naive Bayes
     - Regresión Logística

7. **Evaluación de Modelos**: 
   - Se evalúan los modelos utilizando métricas como Accuracy, F1 Score y Confusion Matrix.
   - Se concluye que el modelo SVM tiene el mejor desempeño con una precisión del 85.45%.

8. **Optimización del Modelo**: 
   - Se realiza una búsqueda en la cuadrícula (Grid Search) para optimizar los hiperparámetros del modelo SVM.

Este notebook proporciona una guía completa desde la carga y preparación de datos hasta la implementación, evaluación y optimización de modelos de Machine Learning para la clasificación de sentimientos en reseñas de películas.

Puedes ver y ejecutar el notebook completo [aquí](https://github.com/FacundoFornaroli/Machine_Learning_Test/blob/d1a7f1f04c8d93818332f4ef44f13ba5cec3517f/IMDB_ML_Prediction.ipynb).
