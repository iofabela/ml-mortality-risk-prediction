"""Load the trained model and score patients.

Usage:
    uv run predict                     # scores the example patient below
    uv run predict patient.json        # scores a patient from a JSON file
"""

import json
import os
import pickle
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd
import xgboost as xgb

from ml_mortality_risk_prediction.features import clean, to_dicts

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "model.bin"))

# A real patient from the test set (id 6785 in data/support2.csv, died in hospital).
EXAMPLE_PATIENT = {
    "age": 59.81,
    "sex": "male",
    "dzgroup": "coma",
    "num_co": 3,
    "edu": 4,
    "income": "under_$11k",
    "scoma": 26,
    "race": "hispanic",
    "hday": 21,
    "diabetes": 1,
    "dementia": 0,
    "ca": "no",
    "dnr_before_adm": 0,
    "meanbp": 105,
    "wblc": 20.1,
    "hrt": 114,
    "resp": 12,
    "temp": 40.59,
    "pafi": 415,
    "alb": 4.3,
    "bili": 0.6,
    "crea": 2.0,
    "sod": 147,
    "ph": 7.64,
    "glucose": 402,
    "bun": 56,
    "urine": 1350,
    "adlsc": 0,
}


@lru_cache(maxsize=1)
def load_model(path: Path = MODEL_PATH) -> dict:
    """Load the artifact written by train.py (cached after the first call)."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `uv run train` first.")
    with open(path, "rb") as f_in:
        return pickle.load(f_in)


def predict_patient(patient: dict, artifact: dict | None = None) -> dict:
    """Return the in-hospital death probability for one patient."""
    artifact = artifact or load_model()
    dv, medians, model = artifact["dv"], artifact["medians"], artifact["model"]

    df = clean(pd.DataFrame([patient]))
    X = dv.transform(to_dicts(df, medians))
    dmatrix = xgb.DMatrix(X, feature_names=list(dv.get_feature_names_out()))
    probability = float(model.predict(dmatrix)[0])

    return {
        "death_probability": round(probability, 4),
        "high_risk": probability >= artifact["threshold"],
        "threshold": artifact["threshold"],
    }


def main() -> None:
    patient = json.loads(Path(sys.argv[1]).read_text()) if len(sys.argv) > 1 else EXAMPLE_PATIENT
    print(json.dumps(predict_patient(patient), indent=2))


if __name__ == "__main__":
    main()
