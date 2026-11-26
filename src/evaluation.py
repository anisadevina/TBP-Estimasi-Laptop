# src/evaluation.py
# ============================================================
# Tahap Evaluasi — 3 langkah:
#   [1] Model Evaluation      → metrik MAE, MSE, RMSE, R² (+MAPE)
#                               pada data test untuk tiap model
#   [2] Model Comparison      → tabel perbandingan ketiga model
#   [3] Best Model Selection  → model terbaik berdasarkan CV RMSE
#                               terendah (kriteria SAMA dengan modeling.py).
#                               Data test TIDAK dipakai untuk memilih model;
#                               ia hanya dipakai melaporkan performa akhir.
#
# Jalankan: python src/evaluation.py
#
# Input :
#   - data/processed/{train,test}.csv + feature_lists.json
#   - results/modeling/models/*.joblib      (hasil src/modeling.py)
#   - results/modeling/model_cv_results.csv (opsional, hasil CV)
#
# Output (results/modeling/):
#   ├── model_evaluation.csv   ← metrik test tiap model
#   ├── model_comparison.csv   ← tabel perbandingan + penanda terbaik (CV)
#   ├── pemenang_per_metrik.csv← model terbaik untuk tiap metrik test
#   ├── prediksi_test.csv      ← prediksi ketiga model vs aktual
#   (best_model.joblib TIDAK ditimpa; dibuat oleh src/modeling.py)
#   └── figures/
#       ├── perbandingan_model.png
#       ├── aktual_vs_prediksi.png
#       └── residual.png
#
# CATATAN: feature importance & analisis error per segmen
#          dikerjakan di tahap Feature Analysis (langkah terpisah).
# ============================================================

import shutil
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.base import RESULTS_DIR, evaluate, load_data

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

MODELS_DIR = RESULTS_DIR / "models"
FIG_DIR    = RESULTS_DIR / "figures"

# Urutan tampil (konsisten dengan src/modeling.py)
MODEL_ORDER = [
    ("multiple_linear_regression.joblib", "Multiple Linear Regression"),
    ("decision_tree.joblib",              "Decision Tree"),
    ("random_forest.joblib",              "Random Forest"),
]

COLORS = ["#4C72B0", "#55A868", "#C44E52"]


# ─────────────────────────────────────────────────────────────
# [1/4] Load data & model
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print(" TBP — Estimasi Harga Laptop | Evaluation")
print("=" * 55)

print("\n[1/4] Membaca data & model terlatih ...")
X_train, X_test, y_train, y_test, NUMERIC, CATEGORICAL = load_data()
print(f"  Train : {X_train.shape}  |  Test : {X_test.shape}")

if not MODELS_DIR.exists():
    sys.exit(f"  ✗ Folder model tidak ditemukan: {MODELS_DIR.resolve()}\n"
             "    Jalankan dulu:  python src/modeling.py")

models = {}          # name -> estimator
model_files = {}     # name -> path file
for fname, name in MODEL_ORDER:
    path = MODELS_DIR / fname
    if not path.exists():
        sys.exit(f"  ✗ Model tidak ditemukan: {path.resolve()}\n"
                 "    Jalankan dulu:  python src/modeling.py")
    models[name] = joblib.load(path)
    model_files[name] = path
    print(f"  ✓ {name:<28} ← {fname}")

# Hasil CV (opsional, untuk tabel perbandingan)
cv_df = None
cv_path = RESULTS_DIR / "model_cv_results.csv"
if cv_path.exists():
    cv_df = pd.read_csv(cv_path)
    print(f"  ✓ CV results ← {cv_path.name}")

# ─────────────────────────────────────────────────────────────
# [2/4] Model Evaluation — metrik pada data test
# ─────────────────────────────────────────────────────────────
print("\n[2/4] Model Evaluation — menghitung metrik pada data test ...")

rows, preds_test, preds_train = [], {}, {}
for name, est in models.items():
    p_test  = est.predict(X_test)
    p_train = est.predict(X_train)
    preds_test[name]  = p_test
    preds_train[name] = p_train

    test_m  = evaluate(y_test, p_test)
    train_m = evaluate(y_train, p_train)
    rows.append({
        "Model":       name,
        "Train_R2":    train_m["R2"],
        "Test_MAE":    test_m["MAE"],
        "Test_MSE":    test_m["MSE"],
        "Test_RMSE":   test_m["RMSE"],
        "Test_R2":     test_m["R2"],
        "Test_MAPE_%": test_m["MAPE_%"],
        "Gap_R2":      round(train_m["R2"] - test_m["R2"], 4),
    })
    print(f"  ✓ {name:<28} MAE={test_m['MAE']:>12,.2f}  "
          f"MSE={test_m['MSE']:>18,.2f}  RMSE={test_m['RMSE']:>12,.2f}  "
          f"R²={test_m['R2']:.4f}")

eval_df = pd.DataFrame(rows)
eval_path = RESULTS_DIR / "model_evaluation.csv"
eval_df.to_csv(eval_path, index=False)
print(f"  → Tersimpan: {eval_path.name}")

# ─────────────────────────────────────────────────────────────
# [3/4] Model Comparison — gabung CV + test, urut CV RMSE
# ─────────────────────────────────────────────────────────────
print("\n[3/4] Model Comparison — menyusun tabel perbandingan ...")

if cv_df is None:
    sys.exit("  ✗ model_cv_results.csv tidak ditemukan (dibutuhkan untuk memilih\n"
             "    model terbaik berdasarkan CV). Jalankan dulu:  python src/modeling.py")

cv_cols = [c for c in ("Model", "CV_RMSE", "CV_RMSE_std", "CV_R2")
           if c in cv_df.columns]
comp_df = eval_df.merge(cv_df[cv_cols], on="Model", how="left")

# Urut berdasarkan CV_RMSE terendah; penanda model terbaik (kriteria = modeling.py)
comp_df = comp_df.sort_values("CV_RMSE").reset_index(drop=True)
best_idx = int(comp_df["CV_RMSE"].idxmin())
comp_df.insert(0, "Terbaik", "-")
comp_df.loc[best_idx, "Terbaik"] = "★"

# Urutan kolom
col_order = (["Terbaik", "Model"]
             + [c for c in ("CV_RMSE", "CV_RMSE_std", "CV_R2") if c in comp_df.columns]
             + ["Train_R2", "Test_MAE", "Test_MSE", "Test_RMSE",
                "Test_R2", "Test_MAPE_%", "Gap_R2"])
comp_df = comp_df[[c for c in col_order if c in comp_df.columns]]

comp_path = RESULTS_DIR / "model_comparison.csv"
comp_df.to_csv(comp_path, index=False)

print("\n  Tabel perbandingan (diurutkan CV_RMSE terendah):")
print("  " + "-" * 100)
print(comp_df.to_string(index=False))
print(f"\n  → Tersimpan: {comp_path.name}")

# Prediksi test (untuk grafik & bahan laporan)
pred_df = pd.DataFrame({"Harga_Aktual": np.asarray(y_test)})
for _fname, name in MODEL_ORDER:
    if name in preds_test:
        pred_df[f"Prediksi_{name}"] = preds_test[name]
pred_path = RESULTS_DIR / "prediksi_test.csv"
pred_df.to_csv(pred_path, index=False)
print(f"  → Tersimpan: {pred_path.name}")

# ─────────────────────────────────────────────────────────────
# [4/4] Best Model Selection — CV RMSE terendah
# ─────────────────────────────────────────────────────────────
print("\n[4/4] Best Model Selection ...")
print("  Kriteria: CV RMSE terendah (5-fold pada data training).")
print("  Data test tidak dipakai untuk memilih; hanya untuk melaporkan performa.")

best_row  = comp_df.loc[best_idx]
best_name = best_row["Model"]

print("\n  " + "-" * 86)
print(f"  {'Model':<28} {'CV_RMSE':>12} {'Test_RMSE':>12} {'Test_MAE':>12} {'Test_R2':>9} {'MAPE_%':>8}")
print("  " + "-" * 86)
for _, r in comp_df.iterrows():
    mark = "★ " if r["Terbaik"] == "★" else "  "
    print(f"{mark}{r['Model']:<28} {r['CV_RMSE']:>12,.2f} {r['Test_RMSE']:>12,.2f} "
          f"{r['Test_MAE']:>12,.2f} {r['Test_R2']:>9.4f} {r['Test_MAPE_%']:>8.2f}")
print("  " + "-" * 86)

# Pemenang tiap metrik (transparansi: metrik bisa memberi pemenang berbeda)
winners = []
for col, better in [("CV_RMSE", "min"), ("Test_MAE", "min"), ("Test_RMSE", "min"),
                    ("Test_R2", "max"), ("Test_MAPE_%", "min")]:
    idx = comp_df[col].idxmin() if better == "min" else comp_df[col].idxmax()
    winners.append({"Metrik": col, "Model_terbaik": comp_df.loc[idx, "Model"],
                    "Nilai": comp_df.loc[idx, col]})
win_df = pd.DataFrame(winners)
win_df.to_csv(RESULTS_DIR / "pemenang_per_metrik.csv", index=False)
print("\n  Pemenang per metrik (RMSE/R² peka terhadap outlier harga):")
print(win_df.to_string(index=False))

# best_model.joblib dibuat oleh modeling.py (kriteria CV). Tidak ditimpa di sini;
# hanya dibuat bila belum ada.
best_out = RESULTS_DIR / "best_model.joblib"
if not best_out.exists():
    shutil.copyfile(model_files[best_name], best_out)
    print(f"\n  → {best_out.name} dibuat ← {model_files[best_name].name}")
else:
    print(f"\n  → {best_out.name} tidak diubah (dibuat oleh modeling.py)")

# ─────────────────────────────────────────────────────────────
# Grafik evaluasi
# ─────────────────────────────────────────────────────────────
print("\n[Grafik] Membuat visualisasi evaluasi ...")
FIG_DIR.mkdir(parents=True, exist_ok=True)
names  = list(models.keys())
colors = COLORS[:len(names)]

# (a) Perbandingan metrik antar model
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
x = np.arange(len(names))
w = 0.35
mae  = [float(comp_df.loc[comp_df["Model"] == n, "Test_MAE"].iloc[0]) for n in names]
rmse = [float(comp_df.loc[comp_df["Model"] == n, "Test_RMSE"].iloc[0]) for n in names]
r2   = [float(comp_df.loc[comp_df["Model"] == n, "Test_R2"].iloc[0]) for n in names]

axes[0].bar(x - w / 2, mae,  w, label="MAE",  color="#4C72B0")
axes[0].bar(x + w / 2, rmse, w, label="RMSE", color="#C44E52")
axes[0].set_xticks(x, names, rotation=12, ha="right")
axes[0].set_ylabel("Error (INR)")
axes[0].set_title("Perbandingan MAE & RMSE (data test)")
axes[0].legend()
axes[0].grid(axis="y", alpha=0.3)

bars = axes[1].bar(names, r2, color=colors, width=0.5)
axes[1].set_xticks(range(len(names)), names, rotation=12, ha="right")
axes[1].set_ylabel("R²")
axes[1].set_ylim(0, 1.05)
axes[1].set_title("Perbandingan R² (data test)")
axes[1].grid(axis="y", alpha=0.3)
for b, v in zip(bars, r2):
    axes[1].text(b.get_x() + b.get_width() / 2, v + 0.02,
                 f"{v:.4f}", ha="center", fontsize=9)

fig.tight_layout()
fig.savefig(FIG_DIR / "perbandingan_model.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  ✓ figures/perbandingan_model.png")

# (b) Aktual vs Prediksi (scatter per model)
fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4.5))
if len(names) == 1:
    axes = [axes]
all_vals = np.concatenate([np.asarray(y_test)] + [preds_test[n] for n in names])
lo, hi = all_vals.min() * 0.95, all_vals.max() * 1.05

for ax, name, c in zip(axes, names, colors):
    yt = np.asarray(y_test)
    yp = preds_test[name]
    ax.scatter(yt, yp, s=18, alpha=0.65, color=c, edgecolors="none")
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="y = x")
    r2_v = float(comp_df.loc[comp_df["Model"] == name, "Test_R2"].iloc[0])
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Harga aktual (INR)")
    ax.set_ylabel("Harga prediksi (INR)")
    ax.set_title(f"{name}\nR² = {r2_v:.4f}")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)

fig.tight_layout()
fig.savefig(FIG_DIR / "aktual_vs_prediksi.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  ✓ figures/aktual_vs_prediksi.png")

# (c) Residual plot (prediksi - aktual) per model
fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4.2))
if len(names) == 1:
    axes = [axes]

for ax, name, c in zip(axes, names, colors):
    yt = np.asarray(y_test)
    resid = preds_test[name] - yt
    ax.scatter(yt, resid, s=18, alpha=0.65, color=c, edgecolors="none")
    ax.axhline(0, color="black", lw=1.2, ls="--")
    ax.set_xlabel("Harga aktual (INR)")
    ax.set_ylabel("Residual (prediksi − aktual)")
    ax.set_title(f"Residual — {name}")
    ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(FIG_DIR / "residual.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  ✓ figures/residual.png")

# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print(" ✅ Evaluasi selesai")
print("=" * 55)
print("File yang dihasilkan:")
print("  • results/modeling/model_evaluation.csv  ← metrik test (MAE, MSE, RMSE, R²)")
print("  • results/modeling/model_comparison.csv  ← tabel perbandingan 3 model")
print("  • results/modeling/prediksi_test.csv     ← prediksi vs aktual")
print("  • results/modeling/pemenang_per_metrik.csv ← pemenang tiap metrik")
print("  • results/modeling/figures/*.png         ← grafik evaluasi")
print(f"\n★  Model terpilih (CV RMSE {best_row['CV_RMSE']:,.2f} INR): {best_name}")
print("\nLangkah selanjutnya: Feature Analysis (feature importance) bila diperlukan.")
