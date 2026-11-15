"""Feature preparation shared by training (train.py) and inference (predict.py).

The same `clean()` runs on the raw SUPPORT2 CSV and on a single patient sent
to the API, so training and serving always see identical features. The steps
match section 1 of notebook/notebook.ipynb.
"""

import numpy as np
import pandas as pd

TARGET = "hospdead"

CATEGORICAL = ["sex", "dzgroup", "dzclass", "race", "ca", "income"]

NUMERICAL = [
    "age", "num_co", "edu", "scoma", "hday", "diabetes", "dementia",
    "meanbp", "wblc", "hrt", "resp", "temp", "pafi", "alb", "bili",
    "crea", "sod", "ph", "glucose", "bun", "urine", "adlsc", "dnr_before_adm",
    "alb_missing", "pafi_missing", "bili_missing", "bun_missing", "urine_missing",
]

# The disease class is fully determined by the disease group.
DZGROUP_TO_DZCLASS = {
    "arf/mosf_w/sepsis": "arf/mosf",
    "mosf_w/malig": "arf/mosf",
    "chf": "copd/chf/cirrhosis",
    "copd": "copd/chf/cirrhosis",
    "cirrhosis": "copd/chf/cirrhosis",
    "lung_cancer": "cancer",
    "colon_cancer": "cancer",
    "coma": "coma",
}

# Fill values recommended by the SUPPORT authors: a lab that was not ordered
# was usually judged to be normal.
NORMAL_VALUES = {
    "alb": 3.5,
    "pafi": 333.3,
    "bili": 1.01,
    "crea": 1.01,
    "bun": 6.51,
    "wblc": 9.0,
    "urine": 2502.0,
}
MISSING_FLAGS = ["alb", "pafi", "bili", "bun", "urine"]

# Raw inputs a patient record can have (dzclass and the *_missing flags are derived).
RAW_INPUTS = (
    ["sex", "dzgroup", "race", "ca", "income", "dnr_before_adm"]
    + [c for c in NUMERICAL if not c.endswith("_missing") and c != "dnr_before_adm"]
)


def normalize(series: pd.Series) -> pd.Series:
    """Lowercase a text column and replace spaces with underscores ("Lung Cancer" -> "lung_cancer")."""
    return series.map(lambda v: v.strip().lower().replace(" ", "_") if isinstance(v, str) else np.nan)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw SUPPORT2 rows (CSV or API input) into the model's feature columns.

    This does not learn anything from the data. Medians for the remaining
    NaNs are learned in training and applied in `to_dicts`.
    """
    df = df.copy()
    df.columns = df.columns.str.lower().str.replace(".", "_")

    # Fields left out of an API/JSON input are treated as unknown.
    for c in RAW_INPUTS:
        if c not in df.columns:
            df[c] = np.nan

    # Only a DNR order placed before admission is known at prediction time.
    if "dnr" in df.columns:
        df["dnr_before_adm"] = (normalize(df["dnr"]) == "dnr_before_sadm").astype(int)
    df["dnr_before_adm"] = pd.to_numeric(df["dnr_before_adm"]).fillna(0).astype(int)

    for c in ["sex", "dzgroup", "race", "ca", "income"]:
        df[c] = normalize(df[c]).fillna("unknown")
    df["dzclass"] = df["dzgroup"].map(DZGROUP_TO_DZCLASS).fillna("unknown")

    base_numerical = [c for c in NUMERICAL if not c.endswith("_missing") and c != "dnr_before_adm"]
    for c in base_numerical:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    # Physiologically impossible values are data-entry errors.
    df.loc[df["alb"] > 10, "alb"] = np.nan
    df.loc[df["glucose"] < 10, "glucose"] = np.nan

    for c in MISSING_FLAGS:
        df[c + "_missing"] = df[c].isnull().astype(int)
    df = df.fillna(NORMAL_VALUES)

    columns = CATEGORICAL + NUMERICAL + ([TARGET] if TARGET in df.columns else [])
    return df[columns]


def to_dicts(df: pd.DataFrame, medians: pd.Series) -> list[dict]:
    """Fill the remaining numeric NaNs with the training medians and convert to records for DictVectorizer."""
    df = df[CATEGORICAL + NUMERICAL].copy()
    df[NUMERICAL] = df[NUMERICAL].fillna(medians)
    return df.to_dict(orient="records")
