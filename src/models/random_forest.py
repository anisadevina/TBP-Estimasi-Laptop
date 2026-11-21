# src/models/random_forest.py
# ============================================================
# Model 3: Random Forest Regressor
# ============================================================
# Jalankan mandiri :  python src/models/random_forest.py
# Atau lewat       :  python src/modeling.py
# ============================================================

import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.base import (RANDOM_STATE, RESULTS_DIR,
                          build_preprocessor, evaluate,
                          load_data, print_summary, save_model)

MODEL_NAME = "Random Forest"
SCALE      = False         # Random Forest tidak butuh scaling
CV_FOLDS   = 5

# ── Hyperparameter grid ───────────────────────────────────────────────────────
PARAM_GRID = {
    "model__n_estimators":    [200, 400],
    "model__max_depth":       [None, 15],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features":    [0.5, "sqrt"],
}
# Untuk pengembangan:
#   - Tambahkan "model__n_estimators": [600, 800] untuk akurasi lebih tinggi
#   - Tambahkan "model__bootstrap": [True, False]
#   - Tambahkan "model__max_samples": [0.6, 0.8, None] (subsample rows)
#   - Gunakan RandomizedSearchCV untuk ruang pencarian lebih luas:
#       from sklearn.model_selection import RandomizedSearchCV
# ─────────────────────────────────────────────────────────────────────────────


def train(X_train, X_test, y_train, y_test,
          numeric_features, categorical_features) -> dict:
    """
    Latih Random Forest Regressor, tuning dengan GridSearchCV.

    Return
    ------
    dict dengan key: name, estimator, predictions, cv_rmse,
                     train_metrics, test_metrics, best_params
    """
    print(f"\n>>> [{MODEL_NAME}] Training dimulai ...")

    pipe = Pipeline([
        ("prep",  build_preprocessor(numeric_features, categorical_features, SCALE)),
        ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
    ])

    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(
        pipe, PARAM_GRID,
        cv=cv, n_jobs=-1, refit="rmse",
        scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"},
    )
    gs.fit(X_train, y_train)

    idx     = gs.best_index_
    cv_rmse = -gs.cv_results_["mean_test_rmse"][idx]
    p_test  = gs.predict(X_test)
    params  = {k.replace("model__", ""): v for k, v in gs.best_params_.items()}

    print(f"  CV RMSE : {cv_rmse:>12,.0f} INR")
    print(f"  Best params : {params}")

    path = save_model(gs.best_estimator_, MODEL_NAME)
    print(f"  Model disimpan → {path}")

    return {
        "name":        MODEL_NAME,
        "estimator":   gs.best_estimator_,
        "predictions": p_test,
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

    # Simpan feature importance bawaan Random Forest
    pipe  = result["estimator"]
    rf    = pipe.named_steps["model"]
    names = pipe.named_steps["prep"].get_feature_names_out()
    fi_df = pd.DataFrame({"fitur": names, "importance": rf.feature_importances_})
    fi_df = fi_df.sort_values("importance", ascending=False)
    out_dir = RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    fi_df.to_csv(out_dir / "feature_importance_rf.csv", index=False)
    print("\nTop 10 feature importance Random Forest:")
    print(fi_df.head(10).round(4).to_string(index=False))
