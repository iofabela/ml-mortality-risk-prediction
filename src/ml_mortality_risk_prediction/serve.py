"""FastAPI web service for the in-hospital mortality model.

Usage:
    uv run serve                       # http://localhost:9696
    API reference (Scalar): http://localhost:9696/docs
"""

import os
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator
from scalar_fastapi import get_scalar_api_reference

from ml_mortality_risk_prediction.features import DZGROUP_TO_DZCLASS
from ml_mortality_risk_prediction.predict import EXAMPLE_PATIENT, load_model, predict_patient

PORT = int(os.getenv("PORT", 9696))


def _normalize(value):
    """Accept dataset spellings too, e.g. "ARF/MOSF w/Sepsis" -> "arf/mosf_w/sepsis"."""
    return value.strip().lower().replace(" ", "_") if isinstance(value, str) else value


class Patient(BaseModel):
    """Patient data collected around day 3 of the hospital stay. Unknown lab values can be null."""

    model_config = ConfigDict(json_schema_extra={"examples": [EXAMPLE_PATIENT]})

    # Demographics
    age: float = Field(ge=18, le=110, description="Age in years")
    sex: Literal["male", "female"]
    race: Literal["white", "black", "hispanic", "asian", "other", "unknown"] = "unknown"
    edu: float | None = Field(None, ge=0, le=31, description="Years of education")
    income: Literal["under_$11k", "$11-$25k", "$25-$50k", ">$50k", "unknown"] = "unknown"

    # Diagnosis and comorbidities
    dzgroup: Literal[tuple(DZGROUP_TO_DZCLASS)] = Field(description="Disease group")
    num_co: int = Field(ge=0, le=9, description="Number of comorbidities")
    diabetes: Literal[0, 1] = 0
    dementia: Literal[0, 1] = 0
    ca: Literal["no", "yes", "metastatic"] = Field("no", description="Cancer status")

    # Day-3 physiology
    scoma: float | None = Field(None, ge=0, le=100, description="SUPPORT coma score (0 = awake)")
    meanbp: float | None = Field(None, ge=0, le=250, description="Mean arterial blood pressure")
    hrt: float | None = Field(None, ge=0, le=300, description="Heart rate")
    resp: float | None = Field(None, ge=0, le=100, description="Respiration rate")
    temp: float | None = Field(None, ge=25, le=45, description="Temperature (Celsius)")
    wblc: float | None = Field(None, ge=0, description="White blood cell count (thousands)")
    pafi: float | None = Field(None, ge=0, description="PaO2/FiO2 ratio")
    alb: float | None = Field(None, ge=0, description="Serum albumin (g/dL)")
    bili: float | None = Field(None, ge=0, description="Bilirubin (mg/dL)")
    crea: float | None = Field(None, ge=0, description="Serum creatinine (mg/dL)")
    sod: float | None = Field(None, ge=90, le=200, description="Serum sodium (mEq/L)")
    ph: float | None = Field(None, ge=6.5, le=8.0, description="Arterial blood pH")
    glucose: float | None = Field(None, ge=0, description="Glucose (mg/dL)")
    bun: float | None = Field(None, ge=0, description="Blood urea nitrogen (mg/dL)")
    urine: float | None = Field(None, ge=0, description="Urine output (mL/day)")

    # Functional status and care context
    adlsc: float | None = Field(None, ge=0, le=7.1, description="Activities of daily living score (0-7)")
    hday: int = Field(1, ge=1, description="Day in hospital at study entry")
    dnr_before_adm: Literal[0, 1] = Field(0, description="1 if a DNR order existed before admission")

    _normalize_text = field_validator("sex", "race", "income", "dzgroup", "ca", mode="before")(
        classmethod(lambda cls, v: _normalize(v))
    )


class Prediction(BaseModel):
    death_probability: float = Field(description="Probability of dying in hospital (0-1)")
    high_risk: bool = Field(description="True if death_probability >= threshold")
    threshold: float = Field(description="Decision threshold, chosen for recall >= 0.80 on validation")


app = FastAPI(
    title="Mortality Risk Prediction",
    description="Predicts the risk of in-hospital death for critically ill adults (SUPPORT2), "
                "using data collected around day 3 of the hospital stay.",
    version="0.1.0",
    docs_url=None,      # Swagger UI is replaced by Scalar at /docs
    redoc_url=None,
)


@app.get("/health")
def health() -> dict:
    """Simple 200 OK check."""
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(patient: Patient) -> Prediction:
    """Return the in-hospital death probability for one patient."""
    try:
        artifact = load_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return Prediction(**predict_patient(patient.model_dump(), artifact))


@app.get("/docs", include_in_schema=False)
def scalar_html():
    """Interactive API reference (Scalar)."""
    return get_scalar_api_reference(openapi_url=app.openapi_url, title=app.title)


def main() -> None:
    uvicorn.run("ml_mortality_risk_prediction.serve:app", host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()
