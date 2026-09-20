# Clasificador de textos por Objetivos de Desarrollo Sostenible (ODS)

Microproyecto 2 del curso **Machine Learning No Supervisado (MLNS)** de la Maestría en Inteligencia Artificial (MAIA), Universidad de los Andes.

**Autores:** Sebastian Guillermo Martinez Velilla y Alvaro Betancourt  

🔗 **Aplicación desplegada:** [https://master-x5jgxjywwfwhgywanawsib.streamlit.app](https://)

## Descripción

La Agenda 2030 de la ONU define 17 Objetivos de Desarrollo Sostenible (ODS). Relacionar textos (informes, artículos, documentos de política pública) con los ODS es una tarea que hoy hacen expertos y consume muchos recursos.

En este proyecto se construyó un modelo que recibe un texto en español y predice con cuál de los ODS se relaciona. Además, se hizo una aplicación en Streamlit donde el usuario escribe un texto y ve el ODS predicho y los 3 ODS con mayor puntaje.

## Datos

Se usó el *OSDG Community Dataset* (OSDG-CD) en su versión traducida al español:

- 9.656 textos, cada uno con su ODS.
- Solo aparecen los ODS del 1 al 16 (el ODS 17 no está en los datos).
- Las clases están desbalanceadas (el ODS 16 tiene unos 1.080 textos y el ODS 12 unos 312).

## Método

El modelo es un `Pipeline` de scikit-learn con estos pasos:

1. **Preprocesamiento:** minúsculas, tokenización, eliminación de stopwords en español y stemming (`SnowballStemmer`).
2. **TF-IDF:** unigramas y bigramas (`ngram_range=(1, 2)`, `min_df=3`, `max_df=0.9`, `max_features=20000`).
3. **Reducción de dimensionalidad:** `TruncatedSVD` con 300 componentes.
4. **Clasificador:** `LinearSVC` con `C=0.5`, elegido con `GridSearchCV` y validación cruzada usando F1 macro.

También se hizo un modelo de tópicos con LSA (`TruncatedSVD` con 15 componentes) para interpretar los temas de los textos frente a los ODS.

## Resultados

Evaluación sobre el 20 % de los textos que no se usó para entrenar:

- **Exactitud:** 88,4 %
- **F1 macro:** ≈ 0,86

Los ODS con mejor desempeño son el 3, 4, 5, 6, 14, 15 y 16. Los que más se confunden entre sí son el 8, 9, 10 y 11, que tratan temas de economía y desarrollo parecidos.

## Estructura del repositorio

```
ODS_Classifier/
  streamlit_app.py            # aplicación de Streamlit
  Definitions.py              # ruta raíz del proyecto
  requirements.txt            # librerías necesarias
  src/
    ModelController.py        # carga el modelo y hace la predicción
    DataPrePreprocessing.py   # función de preprocesamiento de texto
  resources/
    models/
      vectorizer.joblib       # TfidfVectorizer entrenado
      svd.joblib              # TruncatedSVD entrenado
      model.joblib            # LinearSVC entrenado
```

El notebook con todo el desarrollo (exploración, LSA, selección del modelo y evaluación) es `Microproyecto2_ODS_v5.ipynb`.

## Cómo ejecutar la aplicación

1. Clonar el repositorio o descargarlo.
2. Instalar las librerías:
```bash
   cd ODS_Classifier
   pip install -r requirements.txt
```
3. Correr la aplicación:
```bash
   streamlit run streamlit_app.py
```

Los archivos `.joblib` se guardaron con la versión de `scikit-learn` que está en `requirements.txt`, por eso conviene instalar esa misma versión.

## Cómo usar la aplicación

1. Escribir o pegar un texto en español (mínimo 5 palabras).
2. Presionar **Clasificar**.
3. La app muestra el ODS predicho y los 3 ODS con mayor puntaje.

El puntaje es la distancia a la frontera de decisión del `LinearSVC`, no es una probabilidad.

## Limitaciones

- El modelo no puede predecir el ODS 17 porque no está en los datos de entrenamiento.
- Un texto puede tratar varios ODS a la vez, pero el modelo asigna solo uno.
- Funciona con textos en español. Si el texto no tiene palabras que el modelo conozca, la app lo avisa.
- Al usar bolsa de palabras, dos textos con vocabulario parecido pueden confundirse aunque traten temas distintos.
