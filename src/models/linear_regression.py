# src/models/linear_regression.py
# ============================================================
# Model 1: Multiple Linear Regression (baseline)
# ============================================================
# Jalankan mandiri :  python src/models/linear_regression.py
# Atau lewat       :  python src/modeling.py
# ============================================================

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline

# Tambahkan root project ke sys.path agar import base berhasil
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.base import (RANDOM_STATE, RESULTS_DIR,
                          build_preprocessor, evaluate,
                          load_data, print_summary, save_model)

MODEL_NAME  = "Multiple Linear Regression"
SCALE       = True          # Linear Regression butuh scaling
CV_FOLDS    = 5

# ── Hyperparameter grid ───────────────────────────────────────────────────────
# MLR tidak punya hyperparameter yang berarti di-tuning.
# Grid ini ditulis eksplisit agar tetap konsisten dengan format GridSearchCV.
PARAM_GRID = {
    "model__fit_intercept": [True],
}
# Untuk pengembangan: tambahkan regularisasi dengan Ridge/Lasso, misalnya:
#   from sklearn.linear_model import Ridge
#   PARAM_GRID = {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
# ─────────────────────────────────────────────────────────────────────────────


def train(X_train, X_test, y_train, y_test,
          numeric_features, categorical_features) -> dict:
    """
    Latih Multiple Linear Regression, tuning dengan GridSearchCV.

    Return
    ------
    dict dengan key: name, estimator, predictions, cv_rmse,
                     train_metrics, test_metrics, best_params
    """
    print(f"\n>>> [{MODEL_NAME}] Training dimulai ...")

    pipe = Pipeline([
        ("prep",  build_preprocessor(numeric_features, categorical_features, SCALE)),
        ("model", LinearRegression()),
    ])

    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(
        pipe, PARAM_GRID,
        cv=cv, n_jobs=-1, refit="rmse",
        scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"},
    )
    gs.fit(X_train, y_train)

    idx      = gs.best_index_
    cv_rmse  = -gs.cv_results_["mean_test_rmse"][idx]
    p_test   = gs.predict(X_test)
    params   = {k.replace("model__", ""): v for k, v in gs.best_params_.items()}

    print(f"  CV RMSE : {cv_rmse:>12,.0f} INR")
    print(f"  Best params : {params}")

    path = save_model(gs.best_estimator_, MODEL_NAME)
    print(f"  Model disimpan → {path}")

    return {
        "name":        MODEL_NAME,
        "estimator":   gs.best_estimator_,
        "predictions": p_test,          # disimpan agar evaluator bisa pakai langsung
        "cv_rmse":     cv_rmse,
        "cv_rmse_std": gs.cv_results_["std_test_rmse"][idx],
        "cv_r2":       gs.cv_results_["mean_test_r2"][idx],
        "best_params": params,
    }


# ── Jalankan mandiri ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    X_train, X_test, y_train, y_test, NUM, CAT = load_data()
    result = train(X_train, X_test, y_train, y_test, NUM, CAT)

    print("\nTop 10 koefisien (|nilai| terbesar):")
    pipe      = result["estimator"]
    names     = pipe.named_steps["prep"].get_feature_names_out()
    coef_df   = pd.DataFrame({"fitur": names,
                               "koefisien": pipe.named_steps["model"].coef_})
    coef_df   = coef_df.reindex(
        coef_df["koefisien"].abs().sort_values(ascending=False).index
    )
    out_dir = RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    coef_df.to_csv(out_dir / "koefisien_mlr.csv", index=False)
    print(coef_df.head(10).round(2).to_string(index=False))
