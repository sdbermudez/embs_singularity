# CardioAI — Inteligencia Artificial para la Detección de Enfermedades Cardiovasculares

> **Predicción clínica de enfermedad cardíaca mediante Machine Learning sobre el dataset Heart Disease UCI**

---

## Descripción del Proyecto

**CardioAI** es un proyecto de Singularity junto a EMBS aplicado al área médica que utiliza técnicas de **Machine Learning** para predecir la presencia o ausencia de enfermedad cardiovascular en pacientes, a partir de variables clínicas y demográficas.

El modelo entrenado está diseñado para integrarse en un **dashboard interactivo de Streamlit**, donde el clínico o investigador puede ingresar los valores del paciente y obtener en tiempo real la predicción del modelo junto con la probabilidad asociada.

---

## Objetivo

Desarrollar un clasificador binario de alta precisión que, dado un conjunto de variables clínicas de un paciente, determine si existe o no riesgo de enfermedad cardíaca, y desplegarlo en producción como una aplicación web interactiva.

---

## Dataset

Se utilizó el **Heart Disease UCI Dataset**, disponible en el repositorio de Machine Learning de la UCI y en Kaggle. Este dataset consolida datos de cuatro fuentes clínicas distintas:

| Fuente        | País         |
|---------------|--------------|
| Cleveland     | EE.UU.       |
| Hungarian     | Hungría      |
| Switzerland   | Suiza        |
| VA Long Beach | EE.UU.       |

**Tamaño del dataset:** 920 pacientes × 16 variables

### Variables del Dataset

| Variable    | Tipo        | Descripción |
|-------------|-------------|-------------|
| `age`       | Numérica    | Edad del paciente (años) |
| `sex`       | Categórica  | Sexo biológico (Male / Female) |
| `cp`        | Categórica  | Tipo de dolor en el pecho (típico, atípico, no anginal, asintomático) |
| `trestbps`  | Numérica    | Presión arterial en reposo (mm Hg) |
| `chol`      | Numérica    | Colesterol sérico (mg/dl) |
| `fbs`       | Binaria     | Glucosa en ayunas > 120 mg/dl (True/False) |
| `restecg`   | Categórica  | Resultado del electrocardiograma en reposo |
| `thalch`    | Numérica    | Frecuencia cardíaca máxima alcanzada |
| `exang`     | Binaria     | Angina inducida por ejercicio (True/False) |
| `oldpeak`   | Numérica    | Depresión del ST inducida por ejercicio |
| `slope`     | Categórica  | Pendiente del segmento ST en ejercicio máximo |
| `ca`        | Numérica    | Número de vasos principales coloreados por fluoroscopía (0–3) |
| `thal`      | Categórica  | Tipo de defecto de talio |
| `num`       | Target (0/1)| Diagnóstico de enfermedad cardíaca (0 = No, 1 = Sí) |

> **Nota:** Las columnas `id`, `dataset`, `ca` y `thal` fueron eliminadas del modelo final por alta tasa de valores faltantes o no ser relevantes para la predicción.

---

## Metodología

El pipeline del proyecto sigue las etapas estándar de un proyecto de Data Science:

```
Datos Crudos
    ↓
Exploración y Análisis (EDA)
    ↓
Preprocesamiento y Feature Engineering
    ↓
Entrenamiento del Modelo (XGBoost)
    ↓
Evaluación y Validación
    ↓
Exportación del Modelo
    ↓
Despliegue en Streamlit
```

### 1. Exploración de Datos (EDA)
- Análisis de distribución de variables numéricas (histogramas)
- Análisis de relación entre variables numéricas y la variable target (boxplots)
- Tablas de contingencia para variables categóricas vs. target
- Mapa de correlación

### 2. Preprocesamiento
- Codificación de variables categóricas con `cat.codes`
- Eliminación de columnas con >60% de valores faltantes (`ca`, `thal`)
- División en conjuntos de entrenamiento (80%) y prueba (20%)
- El modelo XGBoost maneja internamente los valores faltantes restantes

### 3. Modelo — XGBoost Classifier
Se eligió **XGBoost** por su robustez ante datos tabulares con valores faltantes, su capacidad de regularización y su excelente rendimiento en datasets de tamaño mediano.

**Hiperparámetros principales:**

| Parámetro       | Valor | Descripción |
|-----------------|-------|-------------|
| `n_estimators`  | 200   | Número de árboles |
| `max_depth`     | 4     | Profundidad máxima de los árboles |
| `learning_rate` | 0.05  | Tasa de aprendizaje |
| `subsample`     | 0.8   | Fracción de muestras por árbol |
| `colsample_bytree` | 0.8 | Fracción de features por árbol |
| `eval_metric`   | logloss | Métrica de evaluación interna |

### 4. Importancia de Features
Las variables más influyentes en la predicción son:
1. `cp` (tipo de dolor en el pecho)
2. `thalch` (frecuencia cardíaca máxima)
3. `oldpeak` (depresión del ST)
4. `exang` (angina inducida por ejercicio)
5. `age` (edad)

---


## Despliegue

El modelo se despliega como una **aplicación web interactiva** usando [Streamlit](https://streamlit.io/). La app permite:

- Ingresar los valores clínicos del paciente a través de controles interactivos (sliders, selectboxes)
- Ver la **predicción** del modelo (Enfermedad detectada / No detectada)
- Ver la **probabilidad** asociada a la predicción
- Visualización clara del resultado con semáforo visual

Para instrucciones detalladas de despliegue, consulta el notebook **`streamlit_deploy.ipynb`**.

---

## Stack Tecnológico

| Herramienta     | Versión   | Uso |
|-----------------|-----------|-----|
| Python          | 3.8+      | Lenguaje base |
| pandas          | latest    | Manipulación de datos |
| numpy           | latest    | Operaciones numéricas |
| matplotlib      | latest    | Visualizaciones |
| seaborn         | latest    | Visualizaciones estadísticas |
| scikit-learn    | latest    | Partición de datos, métricas |
| xgboost         | latest    | Modelo de clasificación |
| streamlit       | latest    | Dashboard web interactivo |

---

*Generado con CardioAI — Proyecto de IA Cardiovascular*
