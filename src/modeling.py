# src/modeling.py
# ============================================================
# Orchestrator — Training & Tuning 3 Model
# Jalankan: python src/modeling.py
#
# Atau jalankan model individual:
#   python src/models/linear_regression.py
#   python src/models/decision_tree.py
#   python src/models/random_forest.py
#
# Output: results/modeling/
#   ├── models/            ← .joblib tiap model
#   ├── best_model.joblib  ← model terbaik (CV RMSE terendah)
#   ├── best_params.json   ← hyperparameter terbaik tiap model
#   └── model_cv_results.csv ← hasil CV (bukan test, untuk evaluator)
#
# CATATAN: Evaluasi data test, grafik, dan analisis fitur
#          dikerjakan di tahap Evaluasi (src/evaluation.py).
# ============================================================

import json
import sys
import warnings
from pathlib import Path

import joblib
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.base import RESULTS_DIR, load_data
from models      import train_linear, train_dt, train_rf

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# [1] Load data
# ─────────────────────────────────────────────────────────────
print("\n" + "="*55)
print(" TBP — Estimasi Harga Laptop | Modeling")
print("="*55)
print("\n[1/3] Membaca data hasil preprocessing ...")
X_train, X_test, y_train, y_test, NUMERIC, CATEGORICAL = load_data()
print(f"  Train : {X_train.shape}  |  Test : {X_test.shape}")
print(f"  Fitur numerik     ({len(NUMERIC)}) : {NUMERIC}")
print(f"  Fitur kategorikal ({len(CATEGORICAL)}) : {CATEGORICAL}")

# ─────────────────────────────────────────────────────────────
# [2] Training ketiga model
# ─────────────────────────────────────────────────────────────
print("\n[2/3] Training & tuning model ...")
print("      (GridSearchCV 5-fold, metrik: RMSE)")

results = {}
for train_fn in [train_linear, train_dt, train_rf]:
    r = train_fn(X_train, X_test, y_train, y_test, NUMERIC, CATEGORICAL)
    results[r["name"]] = r

# ─────────────────────────────────────────────────────────────
# [3] Simpan hasil CV + model terbaik
# ─────────────────────────────────────────────────────────────
print("\n[3/3] Menyimpan hasil ...")

# Tabel hasil CV (diserahkan ke evaluator untuk digabung dengan metrik test)
rows = []
for name, r in results.items():
    rows.append({
        "Model":        name,
        "CV_RMSE":      round(r["cv_rmse"], 2),
        "CV_RMSE_std":  round(r["cv_rmse_std"], 2),
        "CV_R2":        round(r["cv_r2"], 4),
        "Best_Params":  str(r["best_params"]),
    })

cv_df = pd.DataFrame(rows).sort_values("CV_RMSE").reset_index(drop=True)
cv_df.to_csv(RESULTS_DIR / "model_cv_results.csv", index=False)

# Simpan best_params.json
best_params_all = {name: r["best_params"] for name, r in results.items()}
with open(RESULTS_DIR / "best_params.json", "w", encoding="utf-8") as fh:
    json.dump(best_params_all, fh, indent=2, default=str, ensure_ascii=False)

# Pilih model terbaik berdasarkan CV RMSE (bukan test)
best_name = cv_df.iloc[0]["Model"]
best_est  = results[best_name]["estimator"]
joblib.dump(best_est, RESULTS_DIR / "best_model.joblib")

print("\n=== HASIL CROSS VALIDATION ===")
print(cv_df[["Model", "CV_RMSE", "CV_RMSE_std", "CV_R2"]].to_string(index=False))
print(f"\n★  Model terbaik (CV RMSE terendah): {best_name}")
print(f"\n✅ Selesai. Output tersimpan di: {RESULTS_DIR.resolve()}")
print("\nFile yang dihasilkan:")
print("  • results/modeling/models/     ← model .joblib tiap algoritma")
print("  • results/modeling/best_model.joblib")
print("  • results/modeling/best_params.json")
print("  • results/modeling/model_cv_results.csv")
print("\nLangkah selanjutnya: jalankan src/evaluation.py (Anggota 3)")
