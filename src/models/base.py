# src/models/base.py
# Utilitas bersama untuk semua model:
#   - load_data()          → load train/test + feature lists
#   - build_preprocessor() → ColumnTransformer dengan/tanpa scaling
#   - evaluate()           → dict MAE, RMSE, R², MAPE
#   - save_results()       → simpan prediksi + joblib

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR     = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR  = PROJECT_ROOT / "results" / "modeling"
RANDOM_STATE = 42


# ─────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────
def load_data() -> tuple:
    """
    Return: X_train, X_test, y_train, y_test, NUMERIC, CATEGORICAL
    """
    for f in ("train.csv", "test.csv", "feature_lists.json"):
        if not (DATA_DIR / f).exists():
            raise FileNotFoundError(
                f"{f} tidak ditemukan di {DATA_DIR.resolve()}.\n"
                "Jalankan preprocessing dulu:  python src/preprocessing.py"
            )

    train = pd.read_csv(DATA_DIR / "train.csv")
    test  = pd.read_csv(DATA_DIR / "test.csv")

    with open(DATA_DIR / "feature_lists.json", encoding="utf-8") as fh:
        feats = json.load(fh)

    NUMERIC     = feats["numeric"]
    CATEGORICAL = feats["categorical"]
    FEATURES    = NUMERIC + CATEGORICAL

    X_train = train[FEATURES]
    y_train = train["Price"]
    X_test  = test[FEATURES]
    y_test  = test["Price"]

    return X_train, X_test, y_train, y_test, NUMERIC, CATEGORICAL


# ─────────────────────────────────────────────────────────────
# Preprocessor builder
# ─────────────────────────────────────────────────────────────
def build_preprocessor(numeric_features: list,
                        categorical_features: list,
                        scale_numeric: bool = True) -> ColumnTransformer:
    """
    Buat ColumnTransformer:
      - numeric  : imputasi median  [+ StandardScaler jika scale_numeric=True]
      - categoric: imputasi modus   + OneHotEncoder
    Di-fit hanya pada data training di dalam pipeline → tidak ada data leakage.
    """
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    num_pipe = Pipeline(steps=num_steps)
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(transformers=[
        ("num", num_pipe, numeric_features),
        ("cat", cat_pipe, categorical_features),
    ])


# ─────────────────────────────────────────────────────────────
# Evaluasi
# ─────────────────────────────────────────────────────────────
def evaluate(y_true, y_pred) -> dict:
    """Hitung MAE, RMSE, R², MAPE dari prediksi vs aktual."""
    return {
        "MAE":    round(mean_absolute_error(y_true, y_pred), 2),
        "RMSE":   round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "R2":     round(r2_score(y_true, y_pred), 4),
        "MAPE_%": round(mean_absolute_percentage_error(y_true, y_pred) * 100, 2),
    }


# ─────────────────────────────────────────────────────────────
# Simpan model & prediksi
# ─────────────────────────────────────────────────────────────
def save_model(model, name: str, out_dir: Path = None) -> Path:
    """Simpan model ke .joblib, return path."""
    out_dir = out_dir or (RESULTS_DIR / "models")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name.lower().replace(' ', '_')}.joblib"
    joblib.dump(model, path)
    return path


def print_summary(name: str, cv_rmse: float, train_metrics: dict, test_metrics: dict):
    """Print ringkasan hasil model ke console."""
    print(f"\n{'='*55}")
    print(f" {name}")
    print(f"{'='*55}")
    print(f"  CV RMSE (rata-rata {5}-fold) : {cv_rmse:>12,.0f} INR")
    print(f"  {'Metrik':<10} {'Train':>12} {'Test':>12}")
    print(f"  {'-'*36}")
    for k in ("MAE", "RMSE", "R2", "MAPE_%"):
        print(f"  {k:<10} {train_metrics[k]:>12,.2f} {test_metrics[k]:>12,.2f}")
