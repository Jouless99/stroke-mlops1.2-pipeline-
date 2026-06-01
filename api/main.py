import os
import logging
from typing import Optional, List

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# ── Configuración ─────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.getenv("MODEL_PATH",    os.path.join(BASE_DIR, "model.pkl"))
BMI_PATH      = os.getenv("BMI_PATH",      os.path.join(BASE_DIR, "bmi_stats.pkl"))
FEATURES_PATH = os.getenv("FEATURES_PATH", os.path.join(BASE_DIR, "feature_names.pkl"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Cargar artefactos al iniciar la API ───────────────────────────────────────
def _load_artifact(path: str, name: str):
    """Carga un artefacto .pkl y lanza un error claro si no se encuentra."""
    if not os.path.exists(path):
        raise RuntimeError(
            f"Artefacto '{name}' no encontrado en: {path}\n"
            "Ejecutá primero el notebook o el DAG de Airflow para generar el modelo."
        )
    logger.info("Cargando %s desde %s", name, path)
    return joblib.load(path)


model         = _load_artifact(MODEL_PATH,    "model")
bmi_stats     = _load_artifact(BMI_PATH,      "bmi_stats")
feature_names = _load_artifact(FEATURES_PATH, "feature_names")

logger.info("Modelo cargado. Features esperadas (%d): %s", len(feature_names), feature_names)

# ── Aplicación FastAPI ────────────────────────────────────────────────────────
app = FastAPI(
    title="Stroke Prediction API",
    description=(
        "API REST para predecir el riesgo de stroke a partir de datos clínicos. "
        "Modelo: Soft Voting Ensemble (Random Forest + XGBoost). "
        "Dataset: Healthcare Stroke Dataset. "
        "Proyecto final MLOps1 – CEIA – FIUBA."
    ),
    version="1.0.0",
    contact={
        "name": "MLOps1 Student",
        "url" : "https://github.com/facundolucianna/amq2-service-ml",
    },
)


# ═══════════════════════════════════════════════════════════════════════════════
# Schemas de entrada y salida
# ═══════════════════════════════════════════════════════════════════════════════

class StrokeInput(BaseModel):
    """
    Datos clínicos del paciente para predicción de stroke.

    Todos los campos categóricos deben enviarse con los valores exactos del dataset original.
    Los campos opcionales pueden omitirse; se imputarán con la mediana del entrenamiento.
    """

    gender: str = Field(
        ...,
        description="Género del paciente. Valores válidos: 'Male', 'Female'.",
        examples=["Male"],
    )
    age: float = Field(
        ...,
        ge=0, le=120,
        description="Edad del paciente en años.",
        examples=[67.0],
    )
    hypertension: int = Field(
        ...,
        ge=0, le=1,
        description="¿El paciente tiene hipertensión? 0 = No, 1 = Sí.",
        examples=[0],
    )
    heart_disease: int = Field(
        ...,
        ge=0, le=1,
        description="¿El paciente tiene enfermedad cardíaca? 0 = No, 1 = Sí.",
        examples=[1],
    )
    ever_married: str = Field(
        ...,
        description="¿El paciente estuvo casado alguna vez? Valores: 'Yes', 'No'.",
        examples=["Yes"],
    )
    work_type: str = Field(
        ...,
        description=(
            "Tipo de trabajo. Valores: 'Private', 'Self-employed', "
            "'Govt_job', 'children', 'Never_worked'."
        ),
        examples=["Private"],
    )
    Residence_type: str = Field(
        ...,
        description="Tipo de residencia. Valores: 'Urban', 'Rural'.",
        examples=["Urban"],
    )
    avg_glucose_level: float = Field(
        ...,
        ge=0,
        description="Nivel promedio de glucosa en sangre (mg/dL).",
        examples=[228.69],
    )
    bmi: Optional[float] = Field(
        default=None,
        ge=0,
        description=(
            "Índice de masa corporal (kg/m²). "
            "Si se omite, se imputará con la mediana del grupo correspondiente."
        ),
        examples=[36.6],
    )
    smoking_status: str = Field(
        ...,
        description=(
            "Estado de fumador. Valores: 'formerly smoked', 'never smoked', "
            "'smokes', 'Unknown'."
        ),
        examples=["formerly smoked"],
    )

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v: str) -> str:
        """Valida que el género sea Male o Female."""
        if v not in ("Male", "Female"):
            raise ValueError("gender debe ser 'Male' o 'Female'")
        return v

    @field_validator("ever_married")
    @classmethod
    def validate_married(cls, v: str) -> str:
        """Valida que ever_married sea Yes o No."""
        if v not in ("Yes", "No"):
            raise ValueError("ever_married debe ser 'Yes' o 'No'")
        return v

    @field_validator("work_type")
    @classmethod
    def validate_work(cls, v: str) -> str:
        """Valida el tipo de trabajo."""
        valid = {"Private", "Self-employed", "Govt_job", "children", "Never_worked"}
        if v not in valid:
            raise ValueError(f"work_type debe ser uno de: {valid}")
        return v

    @field_validator("Residence_type")
    @classmethod
    def validate_residence(cls, v: str) -> str:
        """Valida el tipo de residencia."""
        if v not in ("Urban", "Rural"):
            raise ValueError("Residence_type debe ser 'Urban' o 'Rural'")
        return v

    @field_validator("smoking_status")
    @classmethod
    def validate_smoking(cls, v: str) -> str:
        """Valida el estado de fumador."""
        valid = {"formerly smoked", "never smoked", "smokes", "Unknown"}
        if v not in valid:
            raise ValueError(f"smoking_status debe ser uno de: {valid}")
        return v


class StrokePrediction(BaseModel):
    """Resultado de la predicción de stroke."""
    stroke:      int   = Field(..., description="Predicción: 1 = stroke, 0 = sin stroke.")
    probability: float = Field(..., description="Probabilidad de stroke (0.0 – 1.0).")
    risk_level:  str   = Field(..., description="Nivel de riesgo: 'Bajo', 'Medio', 'Alto'.")


class BatchInput(BaseModel):
    """Lista de pacientes para predicción en lote."""
    patients: List[StrokeInput] = Field(..., description="Lista de pacientes a evaluar.")


class BatchPrediction(BaseModel):
    """Resultados de predicción en lote."""
    predictions: List[StrokePrediction]
    total:       int = Field(..., description="Cantidad total de pacientes evaluados.")
    positive:    int = Field(..., description="Cantidad con predicción de stroke = 1.")


class ModelInfo(BaseModel):
    """Metadatos del modelo cargado."""
    model_type:     str
    features_count: int
    features:       List[str]
    model_path:     str


# ═══════════════════════════════════════════════════════════════════════════════
# Lógica de preprocesamiento para inferencia
# ═══════════════════════════════════════════════════════════════════════════════

def _preprocess_input(patient: StrokeInput) -> pd.DataFrame:
    """
    Convierte un StrokeInput en un DataFrame con las mismas columnas y
    transformaciones que usó el modelo durante el entrenamiento.

    Parameters
    ----------
    patient : StrokeInput
        Datos crudos del paciente recibidos por la API.

    Returns
    -------
    pd.DataFrame
        DataFrame de una fila listo para ser pasado al modelo.
    """
    row = {
        "gender"            : 1 if patient.gender == "Female" else 0,
        "age"               : patient.age,
        "hypertension"      : patient.hypertension,
        "heart_disease"     : patient.heart_disease,
        "ever_married"      : 1 if patient.ever_married == "Yes" else 0,
        "avg_glucose_level" : patient.avg_glucose_level,
        "bmi"               : patient.bmi,
    }

    # One-Hot Encoding de work_type
    for wt in ["Govt_job", "Never_worked", "Private", "Self-employed", "children"]:
        row[f"work_type_{wt}"] = int(patient.work_type == wt)

    # One-Hot Encoding de Residence_type
    for rt in ["Rural", "Urban"]:
        row[f"Residence_type_{rt}"] = int(patient.Residence_type == rt)

    # One-Hot Encoding de smoking_status
    for ss in ["Unknown", "formerly smoked", "never smoked", "smokes"]:
        row[f"smoking_status_{ss}"] = int(patient.smoking_status == ss)

    df_row = pd.DataFrame([row])

    # Imputar BMI si es nulo
    if df_row["bmi"].isnull().any():
        # Sin conocer el label real, se usa la mediana general (promedio de ambas)
        fallback_bmi = (bmi_stats["median_bmi_1"] + bmi_stats["median_bmi_0"]) / 2
        df_row["bmi"] = df_row["bmi"].fillna(fallback_bmi)

    # Aplicar capping IQR
    df_row["bmi"] = df_row["bmi"].clip(bmi_stats["lim_inf"], bmi_stats["lim_sup"])

    # Asegurar que el orden de columnas coincida exactamente con el entrenamiento
    df_row = df_row.reindex(columns=feature_names, fill_value=0)

    return df_row


def _make_prediction(patient: StrokeInput) -> StrokePrediction:
    """
    Genera una predicción de stroke para un único paciente.

    Parameters
    ----------
    patient : StrokeInput
        Datos clínicos del paciente.

    Returns
    -------
    StrokePrediction
        Predicción binaria, probabilidad y nivel de riesgo.
    """
    df_input    = _preprocess_input(patient)
    prediction  = int(model.predict(df_input)[0])
    probability = float(model.predict_proba(df_input)[0][1])

    if probability < 0.30:
        risk_level = "Bajo"
    elif probability < 0.60:
        risk_level = "Medio"
    else:
        risk_level = "Alto"

    return StrokePrediction(
        stroke=prediction,
        probability=round(probability, 4),
        risk_level=risk_level,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@app.get(
    "/",
    summary="Health check",
    description="Verifica que la API está funcionando correctamente.",
    tags=["Status"],
)
def health_check():
    """Retorna el estado de salud de la API."""
    return {"status": "ok", "service": "Stroke Prediction API", "version": "1.0.0"}


@app.get(
    "/info",
    response_model=ModelInfo,
    summary="Información del modelo",
    description="Retorna metadatos del modelo cargado: tipo, features y ruta del artefacto.",
    tags=["Modelo"],
)
def get_model_info():
    """Retorna metadatos del modelo actualmente cargado."""
    return ModelInfo(
        model_type=type(model).__name__,
        features_count=len(feature_names),
        features=feature_names,
        model_path=MODEL_PATH,
    )


@app.post(
    "/predict",
    response_model=StrokePrediction,
    summary="Predecir stroke para un paciente",
    description=(
        "Recibe los datos clínicos de un paciente y retorna la predicción de stroke, "
        "la probabilidad estimada y el nivel de riesgo (Bajo / Medio / Alto)."
    ),
    tags=["Predicción"],
)
def predict(patient: StrokeInput):
    """
    Predice el riesgo de stroke para un único paciente.

    - **stroke = 1**: el modelo predice que el paciente tendrá un stroke.
    - **stroke = 0**: el modelo predice que el paciente no tendrá un stroke.
    - **probability**: probabilidad de stroke en el rango [0, 1].
    - **risk_level**: Bajo (< 30%), Medio (30–60%), Alto (> 60%).
    """
    try:
        return _make_prediction(patient)
    except Exception as e:
        logger.exception("Error durante la predicción: %s", e)
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")


@app.post(
    "/predict/batch",
    response_model=BatchPrediction,
    summary="Predicción en lote",
    description="Recibe una lista de pacientes y retorna la predicción para cada uno.",
    tags=["Predicción"],
)
def predict_batch(batch: BatchInput):
    """
    Predice el riesgo de stroke para múltiples pacientes en una sola llamada.

    Útil para evaluar cohortes de pacientes. El máximo recomendado es 1000 registros.
    """
    try:
        predictions = [_make_prediction(p) for p in batch.patients]
        return BatchPrediction(
            predictions=predictions,
            total=len(predictions),
            positive=sum(p.stroke for p in predictions),
        )
    except Exception as e:
        logger.exception("Error en predicción en lote: %s", e)
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")


# ── Punto de entrada para ejecución directa ───────────────────────────────────
if __name__ == "__main__":
    # Ejecutar con: python main.py
    # O con:        uvicorn main:app --reload --port 8000
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
