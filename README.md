# TBP – Estimasi Harga Laptop

Proyek Data Mining untuk membandingkan algoritma regresi dalam mengestimasi harga laptop berdasarkan spesifikasi hardware.

Harga pada dataset menggunakan **Indian Rupee (INR)**.

## Struktur Proyek

```text
TBP-Estimasi-Laptop/
├── data/
│   ├── raw/
│   │   └── laptop(1).csv
│   └── processed/
├── src/
│   └── preprocessing.py
├── notebooks/
├── models/
├── reports/
├── .gitignore
├── README.md
└── requirements.txt
```

Keterangan:
- `data/raw/`: dataset sumber.
- `data/processed/`: hasil cleaning, feature engineering, pembagian data, dan preprocessing.
- `src/`: kode program utama.
- `notebooks/`: notebook eksplorasi dan eksperimen.
- `models/`: penyimpanan model hasil pelatihan.
- `reports/`: laporan dan dokumentasi hasil penelitian.

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
