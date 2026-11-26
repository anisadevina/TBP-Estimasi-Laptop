# TBP – Estimasi Harga Laptop

Proyek Data Mining untuk membandingkan algoritma regresi dalam mengestimasi harga laptop berdasarkan spesifikasi hardware.

Harga pada dataset menggunakan **Indian Rupee (INR)**.

## Struktur Proyek

```text
TBP-Estimasi-Laptop/
├── data/
│   ├── raw/
│   │   └── laptop(1).csv
│   └── processed/              (dihasilkan preprocessing.py, tidak di-commit)
├── src/
│   ├── preprocessing.py
│   ├── modeling.py             (orchestrator training 3 model)
│   ├── evaluation.py
│   └── models/
│       ├── __init__.py
│       ├── base.py
│       ├── linear_regression.py
│       ├── decision_tree.py
│       └── random_forest.py
├── results/
│   └── modeling/               (model, metrik, grafik, hasil analisis)
├── PANDUAN_PROYEK.md
├── .gitignore
├── README.md
└── requirements.txt
```

Keterangan:
- `data/raw/`: dataset sumber.
- `data/processed/`: hasil cleaning, feature engineering, pembagian data, dan preprocessing.
- `src/`: kode program utama (preprocessing, modeling, evaluasi).
- `results/modeling/`: model terlatih (`.joblib`), hasil CV, metrik test, grafik, dan analisis.

## Persiapan Environment

Pastikan Python sudah terpasang. Buka terminal pada folder utama proyek, kemudian jalankan:

1. Buat virtual environment:

   ```bash
   python -m venv .venv
   ```

2. Aktifkan virtual environment:

   **Windows PowerShell**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   **Windows CMD**
   ```cmd
   .venv\Scripts\activate.bat
   ```

3. Install library:

   ```bash
   pip install -r requirements.txt
   ```

## Menjalankan Preprocessing

Pastikan dataset `laptop(1).csv` tersedia di folder `data/raw/`.

Jalankan perintah berikut dari folder utama proyek:

```bash
python .\src\preprocessing.py
```

Hasil preprocessing akan tersimpan secara otomatis di folder `data/processed/`.

Jika ingin menggunakan dataset atau folder output yang berbeda:

```bash
python .\src\preprocessing.py --data "path/ke/dataset.csv" --output "path/ke/output"
```

## Menjalankan Modeling dan Evaluasi

Jalankan berurutan dari folder utama proyek (setelah preprocessing):

```bash
python src/modeling.py
python src/evaluation.py
```

- `modeling.py` melatih dan menuning 3 algoritma (Multiple Linear Regression, Decision Tree, Random Forest) dengan `GridSearchCV` 5-fold, lalu menyimpan `best_model.joblib` berdasarkan **CV RMSE terendah**.
- `evaluation.py` menghitung MAE, MSE, RMSE, R², dan MAPE pada data test, membuat tabel perbandingan dan grafik. Model terbaik mengikuti kriteria CV yang sama; data test tidak dipakai untuk memilih model.

### Hasil

| Model | CV RMSE | Test MAE | Test RMSE | Test R² | Test MAPE |
|---|---:|---:|---:|---:|---:|
| **Random Forest** (terpilih) | 27.460 | 16.068 | 39.780 | 0,7405 | 17,2% |
| Multiple Linear Regression | 28.957 | 16.888 | 33.482 | 0,8162 | 20,6% |
| Decision Tree | 32.132 | 17.517 | 37.463 | 0,7698 | 19,7% |

Random Forest unggul di CV RMSE, Test MAE, dan MAPE; MLR unggul di Test RMSE dan R² karena RMSE peka terhadap beberapa laptop berharga sangat tinggi. Detail ada di `PANDUAN_PROYEK.md`.

## Output Preprocessing

File yang dihasilkan meliputi:

| File | Keterangan |
|---|---|
| `laptop_clean.csv` | Dataset setelah cleaning |
| `laptop_features.csv` | Dataset final untuk pemodelan |
| `cleaning_log.csv` | Catatan koreksi manual |
| `train.csv` / `test.csv` | Data hasil pembagian train-test |
| `X_train_linear.csv` / `X_test_linear.csv` | Fitur untuk model yang membutuhkan scaling |
| `X_train_tree.csv` / `X_test_tree.csv` | Fitur untuk model berbasis pohon |
| `y_train.csv` / `y_test.csv` | Target harga |
| `preprocessor_linear.joblib` | Preprocessor dengan scaling |
| `preprocessor_tree.joblib` | Preprocessor tanpa scaling |
| `feature_lists.json` | Daftar fitur dan konfigurasi preprocessing |

## Catatan Metodologis

- Target `Price` tetap menggunakan mata uang INR.
- `Rating` dan `Model` tidak digunakan sebagai fitur model.
- Outlier harga tidak dihapus hanya karena terdeteksi menggunakan metode IQR. Laptop premium tetap dipertahankan karena merupakan observasi valid.
- Pembagian data menggunakan `test_size=0.2` dan `random_state=42`.
- Imputer, encoder, dan scaler di-fit hanya pada data training untuk mencegah data leakage.
- Untuk evaluasi cross-validation, gunakan preprocessor di dalam Pipeline bersama estimator dan lakukan cross-validation pada data training mentah.

## Hasil Pengujian Preprocessing

Script telah diuji menggunakan dataset yang terdiri dari 920 baris.

| Tahapan | Hasil |
|---|---:|
| Dataset setelah cleaning | 920 baris |
| Dataset final pemodelan | 920 baris, 17 kolom |
| Data training | 736 baris |
| Data testing | 184 baris |
| Fitur setelah transformasi | 116 fitur |
