"""Train the final model (XGBoost, tuned in notebook/notebook.ipynb) and save it.

Usage:
    uv run train                      # writes models/model.bin
    uv run train --output other.bin
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from ml_mortality_risk_prediction.features import NUMERICAL, TARGET, clean, to_dicts

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "support2.csv"
MODEL_PATH = ROOT / "models" / "model.bin"

RANDOM_STATE = 1
TARGET_RECALL = 0.80

# Best parameters from the tuning in the notebook (section 9.2).
XGB_PARAMS = {
    "eta": 0.05,
    "max_depth": 4,
    "min_child_weight": 30,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "nthread": 8,
    "seed": 1,
    "verbosity": 1,
}
NUM_BOOST_ROUND = 301


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, skipinitialspace=True)
    return clean(df)


def split(df: pd.DataFrame):
    """Make a stratified 60/20/20 split, the same one used in the notebook."""
    df_full_train, df_test = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE, stratify=df[TARGET])
    df_train, df_val = train_test_split(
        df_full_train, test_size=0.25, random_state=RANDOM_STATE, stratify=df_full_train[TARGET])
    return df_full_train, df_train, df_val, df_test


def fit(df: pd.DataFrame):
    """Fit the medians, the DictVectorizer and XGBoost on one dataframe."""
    medians = df[NUMERICAL].median()
    dv = DictVectorizer(sparse=False)
    X = dv.fit_transform(to_dicts(df, medians))
    dtrain = xgb.DMatrix(X, label=df[TARGET].values, feature_names=list(dv.get_feature_names_out()))
    model = xgb.train(XGB_PARAMS, dtrain, num_boost_round=NUM_BOOST_ROUND)
    return dv, medians, model


def score(df: pd.DataFrame, dv, medians, model) -> np.ndarray:
    X = dv.transform(to_dicts(df, medians))
    return model.predict(xgb.DMatrix(X, feature_names=list(dv.get_feature_names_out())))


def choose_threshold(y_true: np.ndarray, y_pred: np.ndarray, target_recall: float = TARGET_RECALL) -> float:
    """Return the highest threshold (0.01 steps) whose recall is still >= target_recall."""
    best = 0.0
    for t in np.linspace(0, 1, 101):
        predicted = y_pred >= t
        recall = (predicted & (y_true == 1)).sum() / (y_true == 1).sum()
        if recall >= target_recall:
            best = t
    return round(float(best), 2)


def report(name: str, y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> None:
    predicted = y_pred >= threshold
    tp = (predicted & (y_true == 1)).sum()
    precision = tp / max(predicted.sum(), 1)
    recall = tp / (y_true == 1).sum()
    print(f"{name:<11} PR-AUC {average_precision_score(y_true, y_pred):.3f} | "
          f"ROC AUC {roc_auc_score(y_true, y_pred):.3f} | "
          f"@{threshold:.2f}: precision {precision:.3f}, recall {recall:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the in-hospital mortality model.")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    args = parser.parse_args()

    df = load_data(args.data)
    df_full_train, df_train, df_val, df_test = split(df)

    # 1. Pick the decision threshold on the validation set.
    dv, medians, model = fit(df_train)
    y_pred_val = score(df_val, dv, medians, model)
    threshold = choose_threshold(df_val[TARGET].values, y_pred_val)
    report("validation", df_val[TARGET].values, y_pred_val, threshold)

    # 2. Retrain on train + validation and evaluate once on the test set.
    dv, medians, model = fit(df_full_train)
    y_pred_test = score(df_test, dv, medians, model)
    report("test", df_test[TARGET].values, y_pred_test, threshold)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {"dv": dv, "medians": medians, "model": model, "threshold": threshold}
    with open(args.output, "wb") as f_out:
        pickle.dump(artifact, f_out)
    print(f"Model saved to {args.output} (threshold {threshold:.2f})")


if __name__ == "__main__":
    main()
