# src/models/decision_tree.py
# ============================================================
# Model 2: Decision Tree Regressor
# ============================================================
# Jalankan mandiri :  python src/models/decision_tree.py
# Atau lewat       :  python src/modeling.py
# ============================================================

import sys
from pathlib import Path

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.base import (RANDOM_STATE, RESULTS_DIR,
                          build_preprocessor, evaluate,
                          load_data, print_summary, save_model)

MODEL_NAME = "Decision Tree"
SCALE      = False         # Decision Tree tidak butuh scaling
CV_FOLDS   = 5

# ── Hyperparameter grid ───────────────────────────────────────────────────────
PARAM_GRID = {
    "model__max_depth":         [3, 5, 7, 10, 15, None],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf":  [1, 2, 4, 8],
}
# Untuk pengembangan:
#   - Tambahkan "model__criterion": ["squared_error", "friedman_mse"]
#   - Tambahkan "model__max_features": [None, "sqrt", 0.5]
#   - Gunakan cost-complexity pruning: "model__ccp_alpha": [0.0, 0.01, 0.1]
# ─────────────────────────────────────────────────────────────────────────────


def train(X_train, X_test, y_train, y_test,
          numeric_features, categorical_features) -> dict:
    """
    Latih Decision Tree Regressor, tuning dengan GridSearchCV.

    Return
    ------
    dict dengan key: name, estimator, predictions, cv_rmse,
                     train_metrics, test_metrics, best_params
    """
    print(f"\n>>> [{MODEL_NAME}] Training dimulai ...")

    pipe = Pipeline([
        ("prep",  build_preprocessor(numeric_features, categorical_features, SCALE)),
        ("model", DecisionTreeRegressor(random_state=RANDOM_STATE)),
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

    # Simpan feature importance bawaan Decision Tree
    import pandas as pd
    pipe   = result["estimator"]
    dt     = pipe.named_steps["model"]
    names  = pipe.named_steps["prep"].get_feature_names_out()
    fi_df  = pd.DataFrame({"fitur": names, "importance": dt.feature_importances_})
    fi_df  = fi_df.sort_values("importance", ascending=False)
    out_dir = RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    fi_df.to_csv(out_dir / "feature_importance_dt.csv", index=False)
    print("\nTop 10 feature importance Decision Tree:")
    print(fi_df.head(10).round(4).to_string(index=False))
