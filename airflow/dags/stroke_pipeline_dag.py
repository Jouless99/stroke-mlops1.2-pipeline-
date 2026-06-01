"""
DAG: stroke_pipeline
====================
Orquesta el ciclo de vida del modelo de predicción de stroke:

    1. validate_data   – verifica que el CSV existe y tiene las columnas esperadas.
    2. preprocess_data – aplica preprocesamiento y guarda los artefactos intermedios.
    3. train_model     – entrena Random Forest + XGBoost con búsqueda de hiperparámetros
                         y registra los runs en MLflow.
    4. evaluate_model  – carga el mejor modelo del experimento MLflow y reporta métricas.
    5. export_model    – serializa el mejor modelo como .pkl para que la API lo consuma.

Modo de ejecución: LOCAL (sin Docker ni servidor MLflow externo).
    - MLflow guarda los runs en  ./mlruns/  relativo al directorio del DAG.
    - Los artefactos intermedios se guardan en /tmp/stroke_mlops/.

Para usar con Docker / servidor MLflow remoto:
    - Cambiar MLFLOW_URI en la sección de configuración.
    - Cambiar DATA_PATH para apuntar al bucket S3/MinIO correspondiente.
"""

import os
import json
import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# ── Configuración ─────────────────────────────────────────────────────────────
# MODIFICAR estas rutas según tu entorno.
DATA_PATH      = os.getenv("STROKE_DATA_PATH", "/opt/airflow/dags/data/healthcare-dataset-stroke-data.csv")
ARTIFACTS_DIR  = os.getenv("ARTIFACTS_DIR", "/tmp/stroke_mlops")
MLFLOW_URI     = os.getenv("MLFLOW_TRACKING_URI", f"file://{ARTIFACTS_DIR}/mlruns")
EXPERIMENT     = "stroke_prediccion_dag"

REQUIRED_COLS = [
    "id", "gender", "age", "hypertension", "heart_disease",
    "ever_married", "work_type", "Residence_type",
    "avg_glucose_level", "bmi", "smoking_status", "stroke",
]

logger = logging.getLogger(__name__)

# ── Argumentos por defecto del DAG ────────────────────────────────────────────
default_args = {
    "owner"           : "mlops_student",
    "depends_on_past" : False,
    "retries"         : 1,
    "retry_delay"     : timedelta(minutes=5),
    "email_on_failure": False,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Funciones de cada task
# ═══════════════════════════════════════════════════════════════════════════════

def validate_data(**context) -> None:
    """
    Verifica que el archivo CSV existe y contiene las columnas requeridas.

    Raises
    ------
    FileNotFoundError
        Si el CSV no se encuentra en DATA_PATH.
    ValueError
        Si faltan columnas obligatorias en el dataset.
    """
    import pandas as pd

    logger.info("Validando dataset en: %s", DATA_PATH)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset no encontrado en: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, nrows=5)
    missing = set(REQUIRED_COLS) - set(df.columns)

    if missing:
        raise ValueError(f"Columnas faltantes en el dataset: {missing}")

    row_count = sum(1 for _ in open(DATA_PATH)) - 1  # rápido, sin cargar todo
    logger.info("Dataset válido. Filas aproximadas: %d", row_count)

    # Pasar metadata a la siguiente task via XCom
    context["ti"].xcom_push(key="row_count", value=row_count)


def preprocess_data(**context) -> None:
    """
    Carga el CSV, aplica el preprocesamiento completo y guarda los artefactos
    intermedios (X_train, X_test, y_train, y_test, bmi_stats, feature_names)
    en ARTIFACTS_DIR como archivos .pkl y .json.

    El preprocesamiento replica exactamente la lógica del trabajo original de AMq1:
    - Eliminar género 'Other'
    - Encoding binario
    - One-Hot Encoding
    - Imputación BMI con mediana estratificada
    - Capping de outliers IQR
    - Balanceo RandomUnderSampler
    """
    import pandas as pd
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from imblearn.under_sampling import RandomUnderSampler

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df = df[df["gender"] != "Other"].reset_index(drop=True)

    df["gender"]       = (df["gender"] == "Female").astype(int)
    df["ever_married"] = (df["ever_married"] == "Yes").astype(int)
    df = pd.get_dummies(df, columns=["work_type", "Residence_type", "smoking_status"])

    X = df.drop(columns=["stroke"])
    y = df["stroke"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Imputación BMI estratificada
    m1 = X_train.loc[y_train == 1, "bmi"].median()
    m0 = X_train.loc[y_train == 0, "bmi"].median()
    X_train.loc[(X_train["bmi"].isnull()) & (y_train == 1), "bmi"] = m1
    X_train.loc[(X_train["bmi"].isnull()) & (y_train == 0), "bmi"] = m0
    X_test.loc[(X_test["bmi"].isnull())  & (y_test == 1),  "bmi"] = m1
    X_test.loc[(X_test["bmi"].isnull())  & (y_test == 0),  "bmi"] = m0

    # Capping outliers IQR
    Q1, Q3 = X_train["bmi"].quantile(0.25), X_train["bmi"].quantile(0.75)
    IQR    = Q3 - Q1
    lim_inf, lim_sup = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    X_train["bmi"] = X_train["bmi"].clip(lim_inf, lim_sup)
    X_test["bmi"]  = X_test["bmi"].clip(lim_inf, lim_sup)

    for _X in [X_train, X_test]:
        bool_cols = _X.select_dtypes(include="bool").columns
        _X[bool_cols] = _X[bool_cols].astype(int)

    X_train = X_train.drop(columns=["id"])
    X_test  = X_test.drop(columns=["id"])

    feature_names = X_train.columns.tolist()
    bmi_stats = {"median_bmi_1": m1, "median_bmi_0": m0, "lim_inf": lim_inf, "lim_sup": lim_sup}

    # Balanceo
    X_full = X_train._append(X_test)
    y_full = y_train._append(y_test)
    rus = RandomUnderSampler(sampling_strategy="majority", random_state=42)
    X_under, y_under = rus.fit_resample(X_full, y_full)
    X_tr, X_te, y_tr, y_te = train_test_split(X_under, y_under, test_size=0.25, random_state=43)

    # Guardar artefactos
    joblib.dump(X_tr,          f"{ARTIFACTS_DIR}/X_train.pkl")
    joblib.dump(X_te,          f"{ARTIFACTS_DIR}/X_test.pkl")
    joblib.dump(y_tr,          f"{ARTIFACTS_DIR}/y_train.pkl")
    joblib.dump(y_te,          f"{ARTIFACTS_DIR}/y_test.pkl")
    joblib.dump(bmi_stats,     f"{ARTIFACTS_DIR}/bmi_stats.pkl")
    joblib.dump(feature_names, f"{ARTIFACTS_DIR}/feature_names.pkl")

    with open(f"{ARTIFACTS_DIR}/bmi_stats.json", "w") as f:
        json.dump(bmi_stats, f, indent=2)

    logger.info("Preprocesamiento completo. Train: %s | Test: %s", X_tr.shape, X_te.shape)
    context["ti"].xcom_push(key="train_shape", value=str(X_tr.shape))


def train_model(**context) -> None:
    """
    Entrena Random Forest y XGBoost con RandomizedSearchCV y registra
    los experimentos en MLflow. El mejor modelo (por AUC) queda guardado
    en ARTIFACTS_DIR/best_model.pkl para la siguiente task.
    """
    import joblib
    import mlflow
    import mlflow.sklearn
    from mlflow.models.signature import infer_signature
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
    from xgboost import XGBClassifier

    X_train = joblib.load(f"{ARTIFACTS_DIR}/X_train.pkl")
    X_test  = joblib.load(f"{ARTIFACTS_DIR}/X_test.pkl")
    y_train = joblib.load(f"{ARTIFACTS_DIR}/y_train.pkl")
    y_test  = joblib.load(f"{ARTIFACTS_DIR}/y_test.pkl")
    feature_names = joblib.load(f"{ARTIFACTS_DIR}/feature_names.pkl")

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_auc = -1
    best_model = None

    candidates = [
        {
            "name": "Random Forest",
            "estimator": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [90, 100, 130, 200],
                "max_depth": list(range(2, 15)),
                "min_samples_split": list(range(2, 8)),
                "criterion": ["entropy"],
            },
        },
        {
            "name": "XGBoost",
            "estimator": XGBClassifier(eval_metric="logloss", random_state=42),
            "params": {
                "n_estimators": [200, 300],
                "max_depth": [3, 4, 5],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.7, 0.8, 0.9],
            },
        },
    ]

    trained = {}
    for c in candidates:
        search = RandomizedSearchCV(
            estimator=c["estimator"],
            param_distributions=c["params"],
            n_iter=15,
            cv=cv,
            scoring="roc_auc",
            random_state=42,
            n_jobs=-1,
        )
        with mlflow.start_run(run_name=c["name"]) as run:
            search.fit(X_train, y_train)
            y_pred  = search.predict(X_test)
            y_proba = search.predict_proba(X_test)[:, 1]

            metrics = {
                "roc_auc" : round(roc_auc_score(y_test, y_proba), 4),
                "f1_score": round(f1_score(y_test, y_pred), 4),
                "accuracy": round(accuracy_score(y_test, y_pred), 4),
            }
            mlflow.log_params(c["params"])
            mlflow.log_metrics(metrics)
            mlflow.set_tag("model_type", c["name"])
            mlflow.set_tag("features", str(feature_names))

            sig = infer_signature(X_train, search.predict(X_train))
            mlflow.sklearn.log_model(search, "model", signature=sig)

            logger.info("%s | AUC=%.4f | run_id=%s", c["name"], metrics["roc_auc"], run.info.run_id)

            if metrics["roc_auc"] > best_auc:
                best_auc   = metrics["roc_auc"]
                best_model = search

        trained[c["name"]] = search

    # Ensamble con los dos mejores
    ensemble = VotingClassifier(
        estimators=[("rf", trained["Random Forest"]), ("xgb", trained["XGBoost"])],
        voting="soft", n_jobs=-1,
    )
    with mlflow.start_run(run_name="Ensemble RF+XGB"):
        ensemble.fit(X_train, y_train)
        y_pred  = ensemble.predict(X_test)
        y_proba = ensemble.predict_proba(X_test)[:, 1]
        auc     = round(roc_auc_score(y_test, y_proba), 4)
        mlflow.log_metrics({"roc_auc": auc, "f1_score": round(f1_score(y_test, y_pred), 4)})
        mlflow.set_tag("model_type", "Ensemble RF+XGB")
        mlflow.sklearn.log_model(ensemble, "model")

        if auc > best_auc:
            best_model = ensemble

    joblib.dump(best_model, f"{ARTIFACTS_DIR}/best_model.pkl")
    logger.info("Mejor modelo guardado (AUC=%.4f)", best_auc)
    context["ti"].xcom_push(key="best_auc", value=best_auc)


def evaluate_model(**context) -> None:
    """
    Carga el mejor modelo, imprime el classification report completo
    y registra las métricas finales como un run de evaluación en MLflow.
    """
    import joblib
    import mlflow
    from sklearn.metrics import classification_report, roc_auc_score

    X_test  = joblib.load(f"{ARTIFACTS_DIR}/X_test.pkl")
    y_test  = joblib.load(f"{ARTIFACTS_DIR}/y_test.pkl")
    model   = joblib.load(f"{ARTIFACTS_DIR}/best_model.pkl")

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred)
    auc    = roc_auc_score(y_test, y_proba)

    logger.info("=== Classification Report ===\n%s", report)
    logger.info("ROC-AUC: %.4f", auc)

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="Evaluacion Final"):
        mlflow.log_metric("final_roc_auc", auc)
        mlflow.log_text(report, "classification_report.txt")


def export_model(**context) -> None:
    """
    Copia el mejor modelo serializado a la carpeta donde la API FastAPI
    lo buscará. También copia los artefactos de preprocesamiento necesarios.

    Nota: En producción, este paso subiría los archivos a un bucket S3/MinIO.
    """
    import shutil

    # Directorio donde la API busca el modelo
    # MODIFICAR esta ruta si cambiás la ubicación de la API.
    api_dir = os.path.join(os.path.dirname(DATA_PATH), "..", "..", "api")
    api_dir = os.path.normpath(api_dir)
    os.makedirs(api_dir, exist_ok=True)

    for fname in ["best_model.pkl", "bmi_stats.pkl", "feature_names.pkl"]:
        src = f"{ARTIFACTS_DIR}/{fname}"
        dst = os.path.join(api_dir, fname.replace("best_model", "model"))
        shutil.copy2(src, dst)
        logger.info("Exportado: %s → %s", src, dst)

    logger.info("Pipeline completo. Modelo listo para ser servido por FastAPI.")


# ═══════════════════════════════════════════════════════════════════════════════
# Definición del DAG
# ═══════════════════════════════════════════════════════════════════════════════

with DAG(
    dag_id="stroke_pipeline",
    description="Pipeline MLOps de predicción de stroke: validación, preprocesamiento, entrenamiento, evaluación y exportación.",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,   # ejecutar manualmente; cambiar a "@weekly" si se desea periódico
    catchup=False,
    tags=["mlops", "stroke", "clasificacion"],
) as dag:

    t1 = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data,
        doc_md="Verifica que el CSV existe y tiene las columnas requeridas.",
    )

    t2 = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data,
        doc_md="Preprocesa el dataset y guarda artefactos intermedios.",
    )

    t3 = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
        doc_md="Entrena RF y XGBoost con RandomizedSearchCV. Registra en MLflow.",
    )

    t4 = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
        doc_md="Evalúa el mejor modelo y reporta métricas finales.",
    )

    t5 = PythonOperator(
        task_id="export_model",
        python_callable=export_model,
        doc_md="Exporta model.pkl, bmi_stats.pkl y feature_names.pkl a la carpeta de la API.",
    )

    # Dependencias lineales del pipeline
    t1 >> t2 >> t3 >> t4 >> t5
