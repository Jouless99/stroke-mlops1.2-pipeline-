# Stroke MLOps Pipeline

**Materia:** MLOps 1 – CEIA – FIUBA  
**Autora:** Julia Maldonado · a2319  
**Dataset:** [Healthcare Stroke Dataset (Kaggle)](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

---

## Descripción del Problema

El accidente cerebrovascular (stroke) es la segunda causa de muerte a nivel mundial. Este proyecto implementa un pipeline de MLOps completo para predecir el **riesgo de stroke** de un paciente a partir de variables clínicas y demográficas.

El modelo entrenado clasifica si un paciente sufrirá un stroke (clase 1) o no (clase 0), usando un **ensamble de Soft Voting** entre Random Forest y XGBoost, elegido como mejor modelo por ROC-AUC.

---

## Arquitectura del Proyecto

```
stroke_mlops/
├── airflow/
│   └── dags/
│       ├── stroke_pipeline_dag.py   # DAG principal (5 etapas)
│       └── data/
│           └── healthcare-dataset-stroke-data.csv
├── api/
│   ├── main.py                      # API FastAPI de inferencia
│   ├── Dockerfile                   # Imagen Docker de la API
│   └── requirements_api.txt
├── notebooks/
│   └── stroke_mlops_pipeline.ipynb  # Exploración y prototipado
├── docker-compose.yml               # Stack completo (Airflow + MLflow + MinIO + API)
├── requirements.txt                 # Dependencias del proyecto
└── README.md
```

### Stack Tecnológico

| Componente | Herramienta | Versión |
|---|---|---|
| Orquestación | Apache Airflow | 2.9.1 |
| Tracking de experimentos | MLflow | 2.13.0 |
| Almacenamiento de artefactos | MinIO (S3-compatible) | latest |
| Base de datos (Airflow) | PostgreSQL | 15 |
| API de inferencia | FastAPI + Uvicorn | 0.111.0 |
| Modelos | scikit-learn, XGBoost | 1.4.2 / 2.0.3 |
| Balanceo de clases | imbalanced-learn | 0.12.3 |

---

## Pipeline (DAG de Airflow)

El DAG `stroke_pipeline` ejecuta **5 etapas en secuencia**:

```
validate_data → preprocess_data → train_model → evaluate_model → export_model
```

1. **`validate_data`** – Verifica que el CSV existe y contiene las columnas requeridas.  
2. **`preprocess_data`** – Encoding, imputación de BMI, capping de outliers IQR y balanceo con `RandomUnderSampler`. Guarda artefactos en `/tmp/stroke_mlops/`.  
3. **`train_model`** – Entrena Random Forest y XGBoost con `RandomizedSearchCV` (5-fold CV). Registra parámetros, métricas y modelos en MLflow. El mejor modelo (por ROC-AUC) queda en `best_model.pkl`.  
4. **`evaluate_model`** – Carga el mejor modelo, genera el `classification_report` completo y lo registra como artefacto en MLflow.  
5. **`export_model`** – Copia `model.pkl`, `bmi_stats.pkl` y `feature_names.pkl` al volumen compartido con la API.

---

## Requisitos Previos

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (v24+)
- [Docker Compose](https://docs.docker.com/compose/) (incluido en Docker Desktop)
- Git

**Recursos recomendados:** al menos 4 GB de RAM para el stack completo.

---

## Guía de Uso Paso a Paso

### 1. Clonar el repositorio

```bash
git clone <URL_DEL_REPOSITORIO>
cd stroke_mlops
```

### 2. Verificar que el dataset está en su lugar

```bash
ls airflow/dags/data/healthcare-dataset-stroke-data.csv
```

Si no está, descargarlo de [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) y colocarlo en esa ruta.

### 3. Levantar el stack con Docker Compose

```bash
docker compose up -d
```

Este comando levanta todos los servicios:
- Airflow Webserver → http://localhost:8080 (usuario: `admin`, contraseña: `admin`)
- MLflow UI → http://localhost:5000
- MinIO Console → http://localhost:9001 (usuario: `minio_admin`, contraseña: `minio_password`)
- API FastAPI → http://localhost:8000

Esperar ~2 minutos para que todos los servicios inicialicen correctamente.

```bash
# Verificar que todos los servicios están corriendo
docker compose ps
```

### 4. Ejecutar el Pipeline

1. Abrir la UI de Airflow en http://localhost:8080
2. Activar el DAG `stroke_pipeline` (toggle ON).
3. Hacer clic en **"Trigger DAG"** (botón ▶).
4. Monitorear las etapas en la vista de Graph o Grid.

Alternativamente, desde la terminal:

```bash
docker compose exec airflow-scheduler airflow dags trigger stroke_pipeline
```

### 5. Ver experimentos en MLflow

Una vez que el DAG complete la etapa `train_model`, abrir http://localhost:5000 y navegar al experimento `stroke_prediccion_dag` para comparar las métricas de los tres modelos.

### 6. Usar la API de Inferencia

Cuando el DAG complete todas las etapas, la API queda lista para recibir predicciones.

**Documentación interactiva:** http://localhost:8000/docs

**Ejemplo de predicción con `curl`:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "age": 67,
    "hypertension": 0,
    "heart_disease": 1,
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "avg_glucose_level": 228.69,
    "bmi": 36.6,
    "smoking_status": "formerly smoked"
  }'
```

**Respuesta esperada:**

```json
{
  "stroke_prediction": 1,
  "stroke_probability": 0.73,
  "risk_level": "Alto"
}
```

### 7. Apagar el stack

```bash
docker compose down
```

Para también eliminar los volúmenes (datos de experimentos):

```bash
docker compose down -v
```

---

## Ejecución en Modo Local (sin Docker)

Si no se dispone de Docker, el notebook `notebooks/stroke_mlops_pipeline.ipynb` permite ejecutar el experimento directamente en un entorno local.

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar el notebook
jupyter notebook notebooks/stroke_mlops_pipeline.ipynb
```

Los experimentos se guardarán localmente en `notebooks/mlruns/`.

---

## Variables de Entorno Configurables

| Variable | Descripción | Valor por defecto |
|---|---|---|
| `STROKE_DATA_PATH` | Ruta al CSV dentro del contenedor | `/opt/airflow/dags/data/...` |
| `MLFLOW_TRACKING_URI` | URI del servidor MLflow | `http://mlflow:5000` |
| `ARTIFACTS_DIR` | Directorio de artefactos intermedios | `/tmp/stroke_mlops` |
| `MODEL_PATH` | Ruta al modelo en la API | `/app/artifacts/model.pkl` |

---

## Preprocesamiento

El pipeline replica la metodología del trabajo original de AMq1:

1. Eliminación de la única fila con género `Other`.
2. Encoding binario para `gender` (Female=1) y `ever_married` (Yes=1).
3. One-Hot Encoding para `work_type`, `Residence_type` y `smoking_status`.
4. Imputación de `bmi` con mediana estratificada por clase (calculada solo en train).
5. Capping de outliers con límites IQR calculados en train.
6. Balanceo con `RandomUnderSampler` sobre la clase mayoritaria.

---

## Resultados del Modelo

Los tres modelos entrenados y sus métricas típicas (pueden variar entre runs):

| Modelo | ROC-AUC | F1-Score | Accuracy |
|---|---|---|---|
| Random Forest | ~0.84 | ~0.77 | ~0.79 |
| XGBoost | ~0.85 | ~0.78 | ~0.80 |
| **Ensamble RF+XGB** | **~0.86** | **~0.79** | **~0.81** |

El **ensamble de Soft Voting** es seleccionado automáticamente como modelo de producción.
