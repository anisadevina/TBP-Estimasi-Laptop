TBP - ESTIMASI HARGA LAPTOP
==========================

Isi folder:
- TBP_Estimasi_Harga_Laptop_FINAL.py : script preprocessing utama
- laptop(1).csv                     : dataset sumber (sediakan sendiri)
- requirements.txt                 : daftar library Python
- output/                           : hasil preprocessing setelah script dijalankan

Cara menjalankan di VS Code:
1. Pastikan Python sudah terpasang.
2. Buka terminal pada folder proyek.
3. Buat virtual environment (opsional tetapi disarankan):
   python -m venv .venv
4. Aktifkan environment:
   Windows PowerShell: .venv\Scripts\Activate.ps1
   Windows CMD       : .venv\Scripts\activate.bat
5. Install library:
   pip install -r requirements.txt
6. Letakkan laptop(1).csv satu folder dengan file .py.
7. Jalankan:
   python TBP_Estimasi_Harga_Laptop_FINAL.py

Jika nama/lokasi dataset berbeda:
   python TBP_Estimasi_Harga_Laptop_FINAL.py --data "path/ke/dataset.csv"

Untuk menentukan folder output:
   python TBP_Estimasi_Harga_Laptop_FINAL.py --output output

Output yang dihasilkan:
- laptop_clean.csv            : data setelah cleaning + fitur turunan
- laptop_features.csv         : 17 kolom final untuk pemodelan (Price + 16 fitur)
- cleaning_log.csv            : catatan koreksi manual yang dilakukan
- train.csv / test.csv        : data train-test mentah (80:20)
- X_train_linear.csv          : fitur train setelah imputasi, encoding, scaling
- X_test_linear.csv           : fitur test setelah preprocessing linear
- X_train_tree.csv            : fitur train setelah imputasi dan encoding (tanpa scaling)
- X_test_tree.csv             : fitur test untuk model berbasis pohon
- y_train.csv / y_test.csv    : target harga (INR)
- preprocessor_linear.joblib  : preprocessor untuk model yang membutuhkan scaling
- preprocessor_tree.joblib    : preprocessor tanpa scaling
- feature_lists.json          : daftar fitur sebelum/sesudah encoding dan konfigurasi split

Catatan metodologis:
- Target Price tetap dalam INR.
- Rating dan Model tidak dipakai sebagai fitur model.
- Outlier harga tidak dihapus hanya karena terdeteksi IQR; laptop premium tetap merupakan observasi valid.
- Split mengikuti notebook: test_size=0.2 dan random_state=42.
- Imputer, encoder, dan scaler di-fit hanya pada data training untuk mencegah data leakage.
- Untuk evaluasi cross-validation, gunakan preprocessor di dalam Pipeline bersama estimator,
  lalu lakukan cross-validation pada data training mentah. Jangan melakukan CV pada matriks
  yang sudah di-fit menggunakan seluruh data training sebelum pembagian fold.

Script telah diuji dijalankan pada dataset 920 baris yang disediakan. Hasil uji:
- df_clean: 920 baris
- df_model: 920 baris dan 17 kolom
- train/test: 736/184 baris
- hasil transformasi: 116 fitur
