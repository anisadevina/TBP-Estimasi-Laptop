# 📊 Panduan Proyek Data Mining
## Estimasi Harga Laptop — Multiple Linear Regression, Decision Tree, dan Random Forest

---

## 1. Ringkasan Proyek

| Komponen | Keterangan |
|---|---|
| **Tugas** | Data mining, kategori estimasi (regresi: memprediksi nilai numerik) |
| **Kasus** | Estimasi harga laptop berdasarkan spesifikasi |
| **Dataset** | `laptop(1).csv`, 920 baris, kolom spesifikasi (CPU, RAM, storage, layar, GPU, OS, garansi) |
| **Variabel target** | `Price` (rupee India / INR, tidak dikonversi) |
| **Metode** | Multiple Linear Regression, Decision Tree, Random Forest |
| **Kerangka kerja** | CRISP-DM (6 tahap) |
| **Evaluasi** | MAE, RMSE, R², MAPE, dengan 5-fold cross validation |
| **Tools** | Python (pandas, scikit-learn, matplotlib, joblib) |

> **Tujuan:** Membangun dan membandingkan empat model regresi untuk mengestimasi harga laptop, lalu menentukan model dengan kinerja terbaik beserta fitur yang paling memengaruhi harga.

---

## 2. Langkah-langkah Pengerjaan (CRISP-DM)

```
Business Understanding → Data Understanding → Data Preparation → Modeling → Evaluation → Deployment
                                                       ↑_____________________|
                                                              (iterasi)
```

---

### Tahap 1 — Business Understanding

**Yang dikerjakan:**
1. Tulis latar belakang: harga laptop sangat beragam dan dipengaruhi banyak spesifikasi, sehingga perlu model estimasi yang objektif.
2. Tulis rumusan masalah: seberapa akurat keempat algoritma mengestimasi harga laptop, dan algoritma mana yang terbaik.
3. Tentukan tujuan, manfaat, dan batasan (data hanya 920 baris, harga dalam INR).
4. Tetapkan variabel target (`Price`) dan kriteria keberhasilan (RMSE dan MAPE terendah, R² tertinggi).

**📄 Output laporan:** Bab 1 Pendahuluan (latar belakang, rumusan masalah, tujuan, batasan).

---

### Tahap 2 — Data Understanding

**Yang dikerjakan** (`laptop_preprocessing.py` — bagian Data Understanding):
1. Muat dataset, cek dimensi, tipe data, nilai kosong, duplikat, dan jumlah nilai unik per kolom.
2. Deteksi baris yang isinya bergeser dari kolom yang seharusnya (misal RAM tertulis di kolom storage).
3. Buat tabel atribut (nama kolom, tipe, contoh nilai, keterangan).
4. Buat statistik deskriptif dan heatmap korelasi.
5. Lihat distribusi harga (menceng ke kanan), sehingga log-transform dicoba sebagai pembanding.

**📄 Output laporan:** Deskripsi dataset, tabel atribut, statistik deskriptif, grafik EDA, heatmap korelasi.

---

### Tahap 3 — Data Preparation

**Yang dikerjakan** (`laptop_preprocessing.py`):

**A. Cleaning:**
- Buang kolom indeks, rapikan teks
- Ubah `Price` dari teks (format India) ke angka
- Petakan ulang baris yang bergeser berdasarkan pola isi sel
- Perbaiki manual dua baris yang bisa diverifikasi
- Isi kategorik kosong dengan `Unknown`
- Semua keputusan tercatat di `cleaning_log.csv`

**B. Feature engineering** — pecah teks spesifikasi menjadi fitur terstruktur:

| Kolom asal | Fitur yang dihasilkan |
|---|---|
| `Generation` | Merek CPU, seri, generasi, sufiks |
| `Core` | Jumlah core dan thread |
| `Ram` | Kapasitas RAM (GB) dan tipenya |
| `SSD` | Total storage (GB) dan tipenya |
| `Display` | Ukuran layar, resolusi, touchscreen |
| `Graphics` | GPU merek, model, dedicated/integrated, VRAM |
| `OS` | Sistem operasi |
| `Warranty` | Durasi garansi |
| *(tambahan)* | Penanda laptop gaming |

**C. Split data:** 80:20 (736 train, 184 test), terstratifikasi pada kuantil harga.

**D. Pipeline preprocessing:**
- Numerik: imputasi median
- Kategorik: imputasi `Unknown` + One-Hot Encoding
- StandardScaler **hanya** untuk Linear Regression dan KNN
- Semua di-fit hanya pada data train (mencegah data leakage)

> ⚠️ **PENTING:** Preprocessing di-fit **hanya pada data training** untuk mencegah data leakage ke data test.

**📄 Output:** `laptop_clean.csv`, `laptop_features.csv`, `train.csv`, `test.csv`

---

### Tahap 4 — Modeling

**Yang dikerjakan** (`laptop_modeling.py`):

1. Gabungkan preprocessing dan model dalam satu `Pipeline`.
2. Latih empat algoritma dengan `GridSearchCV` (5-fold, metrik RMSE):

| Algoritma | Scaling | Hyperparameter yang dituning |
|---|---|---|
| Multiple Linear Regression | Ya | `fit_intercept` |
| Decision Tree | Tidak | `max_depth`, `min_samples_leaf` |
| Random Forest | Tidak | `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features` |

3. Dua skenario target: **raw** (harga asli) dan **log** (`log(1+harga)`), prediksi selalu dikembalikan ke INR.

**📄 Output:** Daftar model, ruang pencarian parameter, parameter terbaik tiap model.

---

### Tahap 5 — Evaluation

**Hasil perbandingan model** (data test, dalam INR):

| Model | Target | CV RMSE | MAE | RMSE | R² | MAPE |
|---|---|---|---|---|---|---|
| **Random Forest** ⭐ | raw | **24.901** | 12.639 | 30.531 | 0,843 | **11,7%** |
| Random Forest | log | 24.961 | 12.518 | 32.239 | 0,825 | 11,0% |
| Linear Regression | raw | 27.747 | 15.675 | 30.169 | **0,847** | 19,9% |
| Linear Regression | log | 26.809 | 16.229 | 48.375 | 0,606 | 13,3% |
| Decision Tree | raw | 29.424 | 17.036 | 35.311 | 0,790 | 16,4% |
| Decision Tree | log | 31.643 | 18.079 | 38.467 | 0,751 | 17,1% |

> ⭐ **Model terpilih: Random Forest (target raw)** — CV RMSE terbaik (24.901), MAPE 11,7%, R² 0,843.

**Temuan utama:**
- 🏆 **Random Forest (raw)** adalah model terbaik berdasarkan CV RMSE, meleset rata-rata ±11,7%.
- ⚠️ **Linear Regression** punya R² test setara (0,847), tetapi MAPE hampir dua kali lipat — kurang akurat untuk laptop murah.
- ❌ **Target log** tidak membantu; justru merugikan Linear Regression (R² turun dari 0,847 → 0,606).
- 📈 **Semua model overfitting**: R² train 0,94–0,999 vs test 0,79–0,84.
- 🔑 **Fitur paling berpengaruh** (Random Forest): RAM, VRAM GPU, resolusi layar, seri CPU.
- 💸 **Error terbesar** di segmen laptop termahal (MAPE 15,7%) dibanding segmen lain (8–13%).

**📄 Output:** Tabel perbandingan, grafik, analisis error, kesimpulan model terbaik.

---

### Tahap 6 — Deployment

**Yang dikerjakan:**
1. Simpan model terbaik dengan joblib → `best_model.joblib`
2. Buat contoh prediksi pada data baru (`contoh_prediksi.csv`)
3. Tulis rencana penerapan: model sebagai alat bantu estimasi, bukan pengganti harga pasar
4. *(Opsional)* Demo sederhana: form input spesifikasi → estimasi harga

**📄 Output:** Subbab implementasi, contoh prediksi, saran pengembangan.

---

## 3. Pembagian Tugas (5 Anggota)

> Anggota 1 sudah mengerjakan bagian preprocessing. Pembagian berikut menyesuaikan kondisi tersebut.

| Anggota | Fokus | Backlog | Status |
|---|---|---|---|
| **Anggota 1** | Data Understanding + Preprocessing | B-01, B-02, B-03, B-04, B-05, B-06, B-11 | 🔄 Sedang jalan |
| **Anggota 2** | Modeling (3 algoritma) + Simpan model | B-07, B-10, B-17 | 🔲 Belum |
| **Anggota 3** | Evaluasi + Analisis fitur | B-08, B-09 | 🔲 Belum |
| **Anggota 4** | Bab Hasil + Kesimpulan + Slide + Review | B-15, B-16, B-18, B-19, B-21 | 🔲 Belum |
| **Anggota 5** | Proposal (Bab 1 + Tinjauan Pustaka + Metodologi) | B-12, B-13, B-14 | 🔲 Belum |

### Detail Per Anggota

#### 👤 Anggota 1 — Data & Preprocessing
- Muat dataset, cek struktur, deteksi baris bergeser
- Cleaning (konversi harga, perbaikan manual, cleaning_log)
- Feature engineering (CPU, RAM, storage, GPU, layar, OS)
- EDA awal (distribusi harga, boxplot, korelasi)
- Split data 80:20 + pipeline preprocessing
- Heatmap korelasi dan tabel atribut (B-11) untuk bahan laporan

#### 👤 Anggota 2 — Modeling
- Buat `modeling.py` — training 3 algoritma: Multiple Linear Regression, Decision Tree, Random Forest
- Tuning hyperparameter dengan GridSearchCV (5-fold)
- Simpan model terbaik (`best_model.joblib`, `best_params.json`)
- Jalankan ulang script di perangkat sendiri, pastikan output konsisten

#### 👤 Anggota 3 — Evaluasi & Analisis Fitur
- Hitung MAE, RMSE, R², MAPE di data test
- Analisis overfitting (R² train vs test)
- Feature importance + residual plot + error per segmen harga
- Siapkan tabel dan grafik untuk bahan Bab Hasil

#### 👤 Anggota 4 — Laporan Hasil + Slide
- **Tulis Bab Hasil dan Pembahasan** (tabel evaluasi, grafik, interpretasi, analisis error)
- **Tulis Kesimpulan dan Saran** (jawab rumusan masalah, keterbatasan)
- Daftar pustaka dan sitasi
- Slide presentasi (ringkasan alur, hasil, kesimpulan)
- Review akhir laporan (konsistensi angka + format sesuai panduan dosen)

#### 👤 Anggota 5 — Proposal
- **Tulis Bab 1 Pendahuluan** — latar belakang, rumusan masalah, tujuan, batasan
- **Tulis Bab Tinjauan Pustaka** — teori CRISP-DM, regresi, 3 algoritma (Linear Regression, Decision Tree, Random Forest), metrik evaluasi (MAE, RMSE, R², MAPE)
- **Tulis Bab Metodologi** — alur CRISP-DM Tahap 2–5, penjelasan tiap langkah

> [!TIP]
> Semua anggota perlu memahami alur data, metode, dan hasil evaluasi agar bisa menjawab pertanyaan saat presentasi.

---

## 4. Product Backlog

> **Keterangan prioritas:** P1 = wajib, P2 = penting, P3 = opsional. Estimasi dalam jam kerja.

| ID | Item | Deskripsi | Prioritas | Estimasi | Status |
|---|---|---|---|---|---|
| B-01 | Muat dataset dan cek struktur | Dimensi, tipe data, missing, duplikat, nilai unik | P1 | 1 jam | ✅ Selesai |
| B-02 | Deteksi baris bergeser | Cek isi kolom vs pola yang seharusnya | P1 | 2 jam | ✅ Selesai |
| B-03 | Data cleaning | Konversi harga, petakan ulang baris, perbaikan manual, cleaning log | P1 | 4 jam | ✅ Selesai |
| B-04 | Feature engineering | Parsing CPU, RAM, storage, layar, GPU, OS, garansi | P1 | 5 jam | ✅ Selesai |
| B-05 | EDA awal | Distribusi harga, boxplot per RAM/CPU/GPU/storage, korelasi | P2 | 2 jam | ✅ Selesai |
| B-06 | Split data dan pipeline | Split 80:20 stratified, preprocessor linear dan tree | P1 | 2 jam | ✅ Selesai |
| B-07 | Modeling 3 algoritma | Multiple LR, Decision Tree, Random Forest, GridSearchCV 5-fold, skenario raw dan log | P1 | 3 jam | ✅ Selesai |
| B-08 | Evaluasi model | MAE, RMSE, R², MAPE, overfitting gap, pemilihan model terbaik | P1 | 2 jam | ✅ Selesai |
| B-09 | Analisis lanjutan | Feature importance, residual, error per segmen harga | P2 | 2 jam | ✅ Selesai |
| B-10 | Simpan model dan contoh prediksi | `best_model.joblib`, `contoh_prediksi.csv` | P2 | 1 jam | ✅ Selesai |
| B-11 | Heatmap korelasi dan tabel atribut | Dibuat dari `laptop_features.csv` untuk Data Understanding | P1 | 2 jam | 🔲 Belum |
| B-12 | Tulis Bab 1 Pendahuluan | Latar belakang, rumusan masalah, tujuan, batasan | P1 | 3 jam | 🔲 Belum |
| B-13 | Tulis Bab Tinjauan Pustaka | Teori CRISP-DM, regresi, empat algoritma, metrik evaluasi | P1 | 4 jam | 🔲 Belum |
| B-14 | Tulis Bab Metodologi | Dari Tahap 2 sampai 5 pada dokumen ini | P1 | 3 jam | 🔲 Belum |
| B-15 | Tulis Bab Hasil dan Pembahasan | Tabel, grafik, interpretasi, analisis error | P1 | 4 jam | 🔲 Belum |
| B-16 | Tulis Kesimpulan dan Saran | Jawab rumusan masalah, keterbatasan, saran | P1 | 1 jam | 🔲 Belum |
| B-17 | Jalankan ulang seluruh script di perangkat sendiri | Pastikan angka di laporan sama dengan output lokal | P1 | 1 jam | 🔲 Belum |
| B-18 | Daftar pustaka dan sitasi | Referensi algoritma, CRISP-DM, sumber dataset | P2 | 2 jam | 🔲 Belum |
| B-19 | Slide presentasi | Ringkasan alur, hasil, dan kesimpulan | P2 | 3 jam | 🔲 Belum |
| B-20 | Demo prediksi sederhana | Form input spesifikasi lalu estimasi harga | P3 | 4 jam | 🔲 Belum |
| B-21 | Review akhir | Konsistensi angka, penulisan, format sesuai panduan dosen | P1 | 2 jam | 🔲 Belum |

---

## 4. Sprint Plan

| Sprint | Fokus | Item Backlog | Target Akhir Sprint |
|---|---|---|---|
| **Sprint 1** ✅ | Pengolahan data dan model | B-01 s.d. B-10 | Selesai |
| **Sprint 2** | Bahan laporan Bab 1–3 | B-11, B-12, B-13, B-17 | Draf Bab 1–3 dan heatmap |
| **Sprint 3** | Bab hasil dan kesimpulan | B-14, B-15, B-16, B-18 | Draf laporan lengkap |
| **Sprint 4** | Finalisasi | B-19, B-20 (opsional), B-21 | Laporan dan slide siap kumpul |

---

## 5. Definition of Done

Sebuah item dianggap **selesai** bila:

- ✅ Output (file, tabel, atau grafik) sudah dihasilkan dan bisa dibuka.
- ✅ Angka di laporan sama dengan output script yang dijalankan ulang.
- ✅ Langkah dan keputusan (misal alasan memilih split 80:20 atau mencoba target log) sudah dijelaskan di laporan.
- ✅ Sudah dicek oleh minimal satu anggota lain.

---

## 6. Risiko dan Catatan

| Risiko / Catatan | Dampak | Penanganan |
|---|---|---|
| Data hanya 920 baris | Model mudah overfitting, hasil bisa bervariasi | Gunakan cross validation, tuliskan sebagai keterbatasan |
| Sebagian baris dataset bergeser kolomnya | Fitur salah bila tidak dibersihkan | Pemetaan ulang berbasis pola, semua tercatat di `cleaning_log.csv` |
| Beberapa RAM di kolom berbeda dengan judul model | Kemungkinan varian konfigurasi | Hanya ditandai, tidak diubah |
| Harga dalam INR | Tidak langsung relevan untuk rupiah | Tulis sebagai batasan, tidak dikonversi |
| Error besar di laptop mahal | Estimasi segmen atas kurang andal | Bahas di analisis error, usulkan tambah data segmen atas |
| Angka bisa sedikit beda antar versi library | Ketidaksamaan dengan laporan | Jalankan ulang di perangkat sendiri sebelum menulis angka final |

---

## 7. Daftar File Hasil

| File | Isi |
|---|---|
| `src/preprocessing.py` | Data understanding, cleaning, feature engineering, EDA, split dan pipeline |
| `laptop_modeling.py` | Modeling 4 algoritma, evaluasi, grafik, simpan model |
| `output/cleaning_log.csv` | Catatan keputusan cleaning |
| `output/train.csv` | Data latih (736 baris, 80%) |
| `output/test.csv` | Data uji (184 baris, 20%) |
| `output/modeling/model_comparison.csv` | Tabel perbandingan model |
| `output/modeling/figures/` | Grafik perbandingan, aktual vs prediksi, residual, feature importance |
| `output/modeling/best_model.joblib` | Model terbaik (Random Forest, target raw) |
| `output/modeling/best_params.json` | Parameter terbaik tiap model |
| `output/modeling/contoh_prediksi.csv` | Contoh prediksi pada data baru |

---

*Panduan ini dibuat berdasarkan backlog dan langkah pengerjaan proyek Data Mining — Estimasi Harga Laptop.*
