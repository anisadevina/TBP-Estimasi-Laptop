# ============================================================
# TBP - ESTIMASI HARGA LAPTOP
# Preprocessing final untuk dijalankan melalui VS Code / terminal
# Sumber proses cleaning & feature engineering: notebook tim (72 cell)
# Harga tetap dalam Indian Rupee (INR).
# ============================================================

from pathlib import Path
import argparse
import json
import re
import numpy as np
import pandas as pd
import joblib
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)


# Lokasi utama proyek (satu tingkat di atas folder src)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocessing dataset harga laptop"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "laptop(1).csv",
        help="Path dataset CSV"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
        help="Folder output"
    )
    return parser.parse_args()

args = parse_args()

DATA_PATH = args.data
OUT_DIR = args.output

OUT_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset tidak ditemukan: {DATA_PATH.resolve()}"
    )


print("\n[1/6] Membaca dataset")
df = pd.read_csv(DATA_PATH)
print("Ukuran data awal:", df.shape)
print("Kolom:", df.columns.tolist())

# Catatan koreksi manual yang memang diterapkan pada notebook.
cleaning_log = []
def log_fix(index, column, old_value, new_value, reason):
    cleaning_log.append({"index": index, "column": column, "old_value": old_value, "new_value": new_value, "reason": reason})




# -------------------- NOTEBOOK CELL 2 --------------------
# ==================================================
# CELL 3: KONVERSI HARGA MENJADI NUMERIK
# ==================================================

# Membuat salinan dataset agar data asli tetap tersedia
df_clean = df.copy()

# Menghapus kolom indeks yang tidak diperlukan
df_clean = df_clean.drop(columns=['Unnamed: 0'], errors='ignore')

# Membersihkan simbol mata uang dan pemisah ribuan
df_clean['Price'] = (
    df_clean['Price']
    .str.replace('₹', '', regex=False)
    .str.replace(',', '', regex=False)
    .str.strip()
    .astype(float)
)

# Memeriksa hasil konversi
print("=== HASIL KONVERSI HARGA ===")
display(df_clean[['Model', 'Price']].head(10))

print("\nTipe data Price:", df_clean['Price'].dtype)

print("\n=== STATISTIK HARGA ===")
display(df_clean['Price'].describe())



# -------------------- NOTEBOOK CELL 8 --------------------
# ==================================================
# CELL 9: PENANGANAN MISSING VALUES
# ==================================================

# 1. Menghapus kolom Rating
df_clean = df_clean.drop(columns=['Rating'])

# 2. Mengisi missing values pada kolom kategorikal
kolom_missing = ['Display', 'OS', 'Warranty']

for col in kolom_missing:
    df_clean[col] = df_clean[col].fillna('Unknown')

# 3. Memeriksa kembali missing values
print("=== MISSING VALUES SETELAH PENANGANAN ===")

display(
    df_clean.isnull().sum().to_frame(
        name='Jumlah Missing'
    )
)

# 4. Memeriksa ukuran dataset
print("\nUkuran dataset setelah penanganan:")
print("Baris:", df_clean.shape[0])
print("Kolom:", df_clean.shape[1])



# -------------------- NOTEBOOK CELL 9 --------------------
# ==================================================
# CELL 10: EKSTRAKSI RAM DAN SSD
# ==================================================

# Ekstraksi kapasitas RAM (GB)
df_clean['RAM_GB'] = (
    df_clean['Ram']
    .str.extract(r'(\d+(?:\.\d+)?)\s*GB', expand=False)
    .astype(float)
)

# Ekstraksi tipe RAM
df_clean['RAM_Type'] = (
    df_clean['Ram']
    .str.extract(r'(DDR\d+[A-Za-z]*)', expand=False)
)

# Ekstraksi kapasitas SSD
def extract_storage(text):
    match = re.search(r'(\d+(?:\.\d+)?)\s*(TB|GB)', str(text), re.I)

    if match:
        value = float(match.group(1))
        unit = match.group(2).upper()

        if unit == 'TB':
            return value * 1024
        return value

    return np.nan

df_clean['SSD_GB'] = df_clean['SSD'].apply(extract_storage)

# Menampilkan hasil
display(
    df_clean[
        ['Ram', 'RAM_GB', 'RAM_Type', 'SSD', 'SSD_GB']
    ].head(15)
)



# -------------------- NOTEBOOK CELL 10 --------------------
# Memperbaiki ekstraksi tipe RAM
df_clean['RAM_Type'] = (
    df_clean['Ram']
    .str.extract(r'((?:LP)?DDR\d+[A-Za-z]*)', expand=False)
    .str.upper()
)

# Memeriksa hasil
display(
    df_clean[['Ram', 'RAM_GB', 'RAM_Type']].head(20)
)



# -------------------- NOTEBOOK CELL 11 --------------------
# Memeriksa missing values hasil ekstraksi
print("Missing RAM_GB:", df_clean['RAM_GB'].isnull().sum())
print("Missing RAM_Type:", df_clean['RAM_Type'].isnull().sum())
print("Missing SSD_GB:", df_clean['SSD_GB'].isnull().sum())

# Menampilkan data yang gagal diekstraksi
display(
    df_clean[
        df_clean[['RAM_GB', 'SSD_GB']].isnull().any(axis=1)
    ][['Model', 'Ram', 'SSD']]
)



# -------------------- NOTEBOOK CELL 12 --------------------
# Ganti nama kolom agar lebih sesuai
df_clean = df_clean.rename(columns={'SSD_GB': 'Storage_GB'})

# Periksa data yang kapasitas RAM atau penyimpanannya kosong
print("Missing RAM_GB:", df_clean['RAM_GB'].isnull().sum())
print("Missing Storage_GB:", df_clean['Storage_GB'].isnull().sum())

display(
    df_clean[
        df_clean[['RAM_GB', 'Storage_GB']].isnull().any(axis=1)
    ][['Model', 'Ram', 'SSD', 'RAM_GB', 'Storage_GB']]
)



# -------------------- NOTEBOOK CELL 13 --------------------
# Perbaiki kapasitas RAM dari kolom Model jika RAM_GB kosong
mask_ram = df_clean['RAM_GB'].isna()

df_clean.loc[mask_ram, 'RAM_GB'] = (
    df_clean.loc[mask_ram, 'Model']
    .str.extract(r'(\d+(?:\.\d+)?)\s*GB', expand=False)
    .astype(float)
)

# Perbaiki kapasitas penyimpanan dari kolom Model, Ram, atau SSD
def extract_storage_from_row(row):
    candidates = [
        row['SSD'],
        row['Ram'],
        row['Model']
    ]

    for value in candidates:
        text = str(value)

        match = re.search(
            r'(\d+(?:\.\d+)?)\s*(TB|GB)\s*(?:NVMe\s*)?(?:SSD|eMMC|Hard Disk)',
            text,
            re.IGNORECASE
        )

        if match:
            number = float(match.group(1))
            unit = match.group(2).upper()

            return number * 1024 if unit == 'TB' else number

    return np.nan


df_clean['Storage_GB'] = df_clean.apply(
    extract_storage_from_row,
    axis=1
)

# Periksa kembali hasilnya
print("Missing RAM_GB:", df_clean['RAM_GB'].isna().sum())
print("Missing Storage_GB:", df_clean['Storage_GB'].isna().sum())

display(
    df_clean[
        df_clean.index.isin([65, 109, 111, 328, 341])
    ][['Model', 'Ram', 'SSD', 'RAM_GB', 'Storage_GB']]
)



# -------------------- NOTEBOOK CELL 14 --------------------
# Perbaiki RAM berdasarkan informasi pada Model
df_clean.loc[109, 'RAM_GB'] = 8
df_clean.loc[328, 'RAM_GB'] = 16
df_clean.loc[341, 'RAM_GB'] = 4

# Periksa kembali
display(
    df_clean.loc[[65, 109, 111, 328, 341],
                 ['Model', 'Ram', 'RAM_GB', 'Storage_GB']]
)



# -------------------- NOTEBOOK CELL 16 --------------------
# Ekstraksi jumlah core
df_clean['CPU_Cores'] = (
    df_clean['Core']
    .str.extract(r'(\d+)\s*Cores?', expand=False)
)

# Ekstraksi jumlah core dari format seperti Hexa Core, Octa Core, dan Quad Core
core_mapping = {
    'Dual Core': 2,
    'Quad Core': 4,
    'Hexa Core': 6,
    'Octa Core': 8
}

for label, value in core_mapping.items():
    mask = df_clean['Core'].str.contains(label, case=False, na=False)
    df_clean.loc[mask & df_clean['CPU_Cores'].isna(), 'CPU_Cores'] = value

# Ekstraksi jumlah thread
df_clean['CPU_Threads'] = (
    df_clean['Core']
    .str.extract(r'(\d+)\s*Threads?', expand=False)
)

# Ubah menjadi numerik
df_clean['CPU_Cores'] = pd.to_numeric(
    df_clean['CPU_Cores'], errors='coerce'
)

df_clean['CPU_Threads'] = pd.to_numeric(
    df_clean['CPU_Threads'], errors='coerce'
)

# Periksa hasil
display(df_clean[['Core', 'CPU_Cores', 'CPU_Threads']].head(20))

print("Missing CPU_Cores:", df_clean['CPU_Cores'].isna().sum())
print("Missing CPU_Threads:", df_clean['CPU_Threads'].isna().sum())



# -------------------- NOTEBOOK CELL 18 --------------------
# Perbaiki jumlah core yang dapat dipastikan
df_clean.loc[341, 'CPU_Cores'] = 8

# Periksa kembali data yang masih kosong
print("Missing CPU_Cores:", df_clean['CPU_Cores'].isna().sum())

display(
    df_clean[df_clean['CPU_Cores'].isna()][
        ['Model', 'Core', 'CPU_Cores', 'CPU_Threads']
    ]
)



# -------------------- NOTEBOOK CELL 20 --------------------
# Ekstraksi merek/keluarga prosesor dari Model
def extract_cpu_brand(model):
    model = str(model).lower()

    if 'core i' in model:
        return 'Intel Core'
    elif 'ryzen' in model:
        return 'AMD Ryzen'
    elif 'snapdragon' in model:
        return 'Snapdragon'
    elif 'kirin' in model:
        return 'Kirin'
    elif 'celeron' in model:
        return 'Intel Celeron'
    elif 'pentium' in model:
        return 'Intel Pentium'
    elif 'athlon' in model:
        return 'AMD Athlon'
    elif 'apple m' in model or 'macbook' in model:
        return 'Apple'
    else:
        return 'Other'

df_clean['CPU_Brand'] = df_clean['Model'].apply(extract_cpu_brand)

# Periksa hasil
display(
    df_clean.loc[[65, 109, 111, 180, 328, 543],
                 ['Model', 'CPU_Brand', 'CPU_Cores', 'CPU_Threads']]
)



# -------------------- NOTEBOOK CELL 21 --------------------
# Ekstraksi generasi Intel Core
df_clean['CPU_Generation'] = (
    df_clean['Model']
    .str.extract(r'(\d+)(?:st|nd|rd|th)\s+Gen', expand=False)
)

df_clean['CPU_Generation'] = pd.to_numeric(
    df_clean['CPU_Generation'], errors='coerce'
)

# Ekstraksi kelas Intel Core (i3, i5, i7, i9)
df_clean['CPU_Class'] = (
    df_clean['Model']
    .str.extract(r'\bCore\s+(i[3579])\b', expand=False)
)

# Periksa hasil
display(
    df_clean[['Model', 'CPU_Brand', 'CPU_Generation', 'CPU_Class']]
    .head(30)
)

print("Missing CPU_Generation:", df_clean['CPU_Generation'].isna().sum())
print("Missing CPU_Class:", df_clean['CPU_Class'].isna().sum())



# -------------------- NOTEBOOK CELL 22 --------------------
# Ekstraksi kelas prosesor
def extract_cpu_class(model):
    model = str(model)

    patterns = [
        r'Core\s+Ultra\s+([579])',
        r'Core\s+([3579])\s+Processor',
        r'Core\s+i([3579])',
        r'Ryzen\s+([3579])',
    ]

    for pattern in patterns:
        match = re.search(pattern, model, re.IGNORECASE)
        if match:
            if 'Ultra' in pattern:
                return 'Core Ultra ' + match.group(1)
            elif 'Ryzen' in pattern:
                return 'Ryzen ' + match.group(1)
            elif 'Processor' in pattern:
                return 'Core ' + match.group(1)
            else:
                return 'Core i' + match.group(1)

    match = re.search(r'Apple\s+M([1-9])', model, re.IGNORECASE)
    if match:
        return 'Apple M' + match.group(1)

    return np.nan


# Ekstraksi generasi dari format nama CPU
def extract_cpu_generation(model):
    model = str(model)

    # Intel: 11th Gen, 12th Gen, dan seterusnya
    match = re.search(r'(\d+)(?:st|nd|rd|th)\s+Gen', model, re.IGNORECASE)
    if match:
        return int(match.group(1))

    # AMD Ryzen: 5600H, 7535HS, 7840HS, dan sejenisnya
    match = re.search(r'Ryzen\s+[3579]\s+(\d{4})', model, re.IGNORECASE)
    if match:
        return int(match.group(1)[0])

    return np.nan


df_clean['CPU_Class'] = df_clean['Model'].apply(extract_cpu_class)
df_clean['CPU_Generation'] = df_clean['Model'].apply(extract_cpu_generation)

# Periksa hasil
display(
    df_clean[['Model', 'CPU_Brand', 'CPU_Generation', 'CPU_Class']].head(30)
)

print("Missing CPU_Generation:", df_clean['CPU_Generation'].isna().sum())
print("Missing CPU_Class:", df_clean['CPU_Class'].isna().sum())



# -------------------- NOTEBOOK CELL 23 --------------------
# Ekstraksi ulang generasi Intel
def extract_intel_generation(model):
    match = re.search(
        r'(\d+)(?:st|nd|rd|th)\s+Gen',
        str(model),
        re.IGNORECASE
    )

    if match:
        return int(match.group(1))

    return np.nan


# Ekstraksi seri AMD Ryzen
def extract_amd_series(model):
    match = re.search(
        r'Ryzen\s+[3579]\s+(\d{4})',
        str(model),
        re.IGNORECASE
    )

    if match:
        return int(match.group(1))

    return np.nan


df_clean['CPU_Generation'] = df_clean['Model'].apply(
    extract_intel_generation
)

df_clean['AMD_Series'] = df_clean['Model'].apply(
    extract_amd_series
)

# Periksa hasil
display(
    df_clean[
        df_clean['CPU_Brand'].eq('AMD Ryzen')
    ][['Model', 'CPU_Class', 'AMD_Series']].head(20)
)

print(
    "Missing Intel CPU_Generation:",
    df_clean['CPU_Generation'].isna().sum()
)

print(
    "Missing AMD_Series:",
    df_clean['AMD_Series'].isna().sum()
)



# -------------------- NOTEBOOK CELL 26 --------------------
def extract_cpu_class(model):
    model = str(model)

    patterns = [
        (r'Core\s+Ultra\s+([579])', 'Core Ultra '),
        (r'Core\s+([3579])\s+Processor', 'Core '),
        (r'Core\s+i([3579])', 'Core i'),
        (r'Ryzen\s+([3579])', 'Ryzen ')
    ]

    for pattern, prefix in patterns:
        match = re.search(pattern, model, re.IGNORECASE)
        if match:
            return prefix + match.group(1)

    match = re.search(r'Apple\s+M([1-9])', model, re.IGNORECASE)
    if match:
        return 'Apple M' + match.group(1)

    return np.nan


df_clean['CPU_Class'] = df_clean['Model'].apply(extract_cpu_class)

display(df_clean['CPU_Class'].value_counts(dropna=False))



# -------------------- NOTEBOOK CELL 28 --------------------
def extract_cpu_class(model):
    model = str(model)

    patterns = [
        (r'Core\s+Ultra\s+([3579])', 'Core Ultra '),
        (r'Core\s+([3579])\s+Processor', 'Core '),
        (r'Core\s+i([3579])', 'Core i'),
        (r'Ryzen\s+([3579])', 'Ryzen '),
        (r'AMD\s+R([3579])\s', 'Ryzen '),
        (r'AMD\s+R([3579])-', 'Ryzen '),
        (r'Ryzen([3579])-', 'Ryzen '),
    ]

    for pattern, prefix in patterns:
        match = re.search(pattern, model, re.IGNORECASE)
        if match:
            return prefix + match.group(1)

    match = re.search(r'Apple\s+M([1-9])', model, re.IGNORECASE)
    if match:
        return 'Apple M' + match.group(1)

    if re.search(r'Pentium', model, re.IGNORECASE):
        return 'Pentium'

    if re.search(r'Celeron', model, re.IGNORECASE):
        return 'Celeron'

    if re.search(r'Athlon', model, re.IGNORECASE):
        return 'Athlon'

    if re.search(r'Snapdragon', model, re.IGNORECASE):
        return 'Snapdragon'

    if re.search(r'Kirin', model, re.IGNORECASE):
        return 'Kirin'

    if re.search(r'Kompanio|MediaTek|MT8183', model, re.IGNORECASE):
        return 'MediaTek'

    if re.search(r'Apple|AQC', model, re.IGNORECASE):
        return 'Other'

    return np.nan


df_clean['CPU_Class'] = df_clean['Model'].apply(extract_cpu_class)

print("Missing CPU_Class:", df_clean['CPU_Class'].isna().sum())

display(df_clean['CPU_Class'].value_counts(dropna=False))



# -------------------- NOTEBOOK CELL 30 --------------------
# Perbaiki tiga data Dell Inspiron
df_clean.loc[332, 'CPU_Class'] = 'Ryzen 3'
df_clean.loc[612, 'CPU_Class'] = 'Ryzen 5'
df_clean.loc[766, 'CPU_Class'] = 'Ryzen 5'

# Perbaiki klasifikasi merek Dell
df_clean.loc[[332, 612, 766], 'CPU_Brand'] = 'AMD Ryzen'

# Periksa kembali enam data
display(
    df_clean.loc[
        [332, 341, 612, 676, 766, 879],
        ['Model', 'CPU_Brand', 'CPU_Class', 'CPU_Cores', 'CPU_Threads']
    ]
)

print("Missing CPU_Class:", df_clean['CPU_Class'].isna().sum())



# -------------------- NOTEBOOK CELL 33 --------------------
def extract_gpu_brand(text):
    text = str(text).lower()

    if 'nvidia' in text:
        return 'NVIDIA'
    elif 'amd' in text or 'radeon' in text:
        return 'AMD'
    elif 'intel' in text:
        return 'Intel'
    elif 'apple' in text or 'core gpu' in text:
        return 'Apple'
    else:
        return 'Other'


def extract_vram(text):
    match = re.search(r'(\d+(?:\.\d+)?)\s*GB', str(text), re.I)

    if match:
        return float(match.group(1))
    return np.nan


def extract_gpu_model(text):
    text = str(text).lower()

    # NVIDIA
    match = re.search(r'rtx\s*(\d{3,4})\s*(ti)?', text)
    if match:
        return 'RTX ' + match.group(1) + (' Ti' if match.group(2) else '')

    match = re.search(r'gtx\s*(\d{3,4})', text)
    if match:
        return 'GTX ' + match.group(1)

    match = re.search(r'mx\s*(\d+)', text)
    if match:
        return 'MX ' + match.group(1)

    match = re.search(r'quadro\s*([a-z]?\d+)', text)
    if match:
        return 'Quadro ' + match.group(1).upper()

    match = re.search(r'rtx\s*(a\d+)', text)
    if match:
        return 'RTX ' + match.group(1).upper()

    # AMD
    match = re.search(r'rx\s*(\d+)\s*(\d*)\s*(m)?', text)
    if match:
        return 'RX ' + match.group(1) + match.group(2) + ('M' if match.group(3) else '')

    match = re.search(r'radeon\s*(\d+m)', text)
    if match:
        return 'Radeon ' + match.group(1).upper()

    match = re.search(r'radeon\s*(vega\s*\d+)', text)
    if match:
        return match.group(1).title()

    # Intel
    if 'arc' in text:
        match = re.search(r'a\d{3,4}m?', text)
        return 'Arc ' + match.group(0).upper() if match else 'Arc'

    if 'iris xe' in text or 'iris xᵉ' in text:
        return 'Iris Xe'

    if 'uhd' in text:
        return 'UHD Graphics'

    if 'hd graphics' in text:
        return 'HD Graphics'

    if 'intel graphics' in text:
        return 'Intel Graphics'

    # Apple
    match = re.search(r'(\d+)[ -]?core gpu', text)
    if match:
        return match.group(1) + '-Core GPU'

    if 'integrated' in text or 'radeon graphics' in text:
        return 'Integrated Graphics'

    return 'Other'


df_clean['GPU_Brand'] = df_clean['Graphics'].apply(extract_gpu_brand)
df_clean['GPU_Model'] = df_clean['Graphics'].apply(extract_gpu_model)
df_clean['VRAM_GB'] = df_clean['Graphics'].apply(extract_vram)

df_clean['GPU_Type'] = np.where(
    df_clean['GPU_Brand'].isin(['NVIDIA']) |
    df_clean['GPU_Model'].str.startswith(('RX ', 'Arc A'), na=False),
    'Dedicated',
    'Integrated'
)

print("Distribusi GPU Brand:")
display(df_clean['GPU_Brand'].value_counts())

print("\nDistribusi GPU Type:")
display(df_clean['GPU_Type'].value_counts())

print("\nDistribusi GPU Model:")
display(df_clean['GPU_Model'].value_counts())

print("\nMissing VRAM:", df_clean['VRAM_GB'].isna().sum())



# -------------------- NOTEBOOK CELL 36 --------------------
def extract_gpu_brand(text):
    text = str(text).lower()

    if 'nvidia' in text:
        return 'NVIDIA'
    elif 'amd' in text or 'radeon' in text:
        return 'AMD'
    elif 'intel' in text:
        return 'Intel'
    elif 'mali' in text:
        return 'ARM'
    elif 'apple' in text or 'core gpu' in text:
        return 'Apple'
    else:
        return 'Unknown'


def extract_gpu_model(text):
    text = str(text).lower()

    # NVIDIA
    match = re.search(r'rtx\s*(\d{3,4})\s*(ti)?', text)
    if match:
        return 'RTX ' + match.group(1) + (' Ti' if match.group(2) else '')

    match = re.search(r'gtx\s*(\d{3,4})', text)
    if match:
        return 'GTX ' + match.group(1)

    match = re.search(r'mx\s*(\d+)', text)
    if match:
        return 'MX ' + match.group(1)

    if 'nvidia' in text and 't550' in text:
        return 'T550'

    if 'nvidia' in text and 'quadro' in text:
        return 'Quadro'

    if 'nvidia' in text and 'rtx' in text:
        return 'RTX (Unspecified)'

    # AMD Radeon
    match = re.search(r'rx\s*(\d+)\s*(\d*)\s*(m)?', text)
    if match:
        return 'RX ' + match.group(1) + match.group(2) + ('M' if match.group(3) else '')

    match = re.search(r'radeon\s*(\d+m)', text)
    if match:
        return 'Radeon ' + match.group(1).upper()

    match = re.search(r'radeon\s*(vega\s*\d+)', text)
    if match:
        return match.group(1).title()

    if 'radeon' in text or 'amd graphics' in text:
        return 'Radeon (Unspecified)'

    # Intel
    if 'arc' in text:
        match = re.search(r'a\d{3,4}m?', text)
        return 'Arc ' + match.group(0).upper() if match else 'Arc'

    if 'iris x' in text:
        return 'Iris Xe'

    if 'uhd' in text:
        return 'UHD Graphics'

    if 'hd graphics' in text:
        return 'HD Graphics'

    if 'intel graphics' in text:
        return 'Intel Graphics'

    # ARM
    if 'mali' in text:
        match = re.search(r'mali[- ]?([a-z]?\d+)', text)
        return 'Mali ' + match.group(1).upper() if match else 'Mali'

    # Apple
    match = re.search(r'(\d+)[ -]?core gpu', text)
    if match:
        return match.group(1) + '-Core GPU'

    return 'Unknown'


df_clean['GPU_Brand'] = df_clean['Graphics'].apply(extract_gpu_brand)
df_clean['GPU_Model'] = df_clean['Graphics'].apply(extract_gpu_model)

print("Distribusi GPU Brand:")
display(df_clean['GPU_Brand'].value_counts())

print("\nDistribusi GPU Model:")
display(df_clean['GPU_Model'].value_counts())



# -------------------- NOTEBOOK CELL 38 --------------------
def extract_gpu_brand(text):
    text = str(text).lower()

    if 'nvidia' in text:
        return 'NVIDIA'
    elif 'amd' in text or 'radeon' in text:
        return 'AMD'
    elif 'intel' in text:
        return 'Intel'
    elif 'mali' in text:
        return 'ARM'
    elif 'apple' in text or 'core gpu' in text:
        return 'Apple'
    else:
        return 'Unknown'


def extract_gpu_model(text):
    text = str(text).lower()

    # NVIDIA
    match = re.search(r'rtx\s*(\d{3,4})\s*(ti)?', text)
    if match:
        return 'RTX ' + match.group(1) + (' Ti' if match.group(2) else '')

    match = re.search(r'gtx\s*(\d{3,4})', text)
    if match:
        return 'GTX ' + match.group(1)

    match = re.search(r'mx\s*(\d+)', text)
    if match:
        return 'MX ' + match.group(1)

    if 't550' in text:
        return 'T550'

    if 'quadro' in text:
        return 'Quadro'

    if 'nvidia' in text and 'rtx' in text:
        return 'RTX (Unspecified)'

    # AMD
    match = re.search(r'rx\s*(\d+)\s*(\d*)\s*(m)?', text)
    if match:
        return 'RX ' + match.group(1) + match.group(2) + ('M' if match.group(3) else '')

    match = re.search(r'radeon\s*(\d+m)', text)
    if match:
        return 'Radeon ' + match.group(1).upper()

    match = re.search(r'radeon\s*(vega\s*\d+)', text)
    if match:
        return match.group(1).title()

    if 'radeon' in text or 'amd integrated' in text or 'amd graphics' in text:
        return 'Radeon (Unspecified)'

    # Intel
    if 'arc' in text:
        match = re.search(r'a\d{3,4}m?', text)
        return 'Arc ' + match.group(0).upper() if match else 'Arc'

    if 'iris x' in text:
        return 'Iris Xe'

    if 'uhd' in text:
        return 'UHD Graphics'

    if 'hd graphics' in text:
        return 'HD Graphics'

    if 'intel integrated' in text or 'intel graphics' in text:
        return 'Integrated Graphics'

    # ARM
    if 'mali' in text:
        match = re.search(r'mali[- ]?([a-z]?\d+)', text)
        return 'Mali ' + match.group(1).upper() if match else 'Mali'

    # Apple
    match = re.search(r'(\d+)[ -]?core gpu', text)
    if match:
        return match.group(1) + '-Core GPU'

    if 'apple m1 integrated graphics' in text:
        return 'Integrated Graphics'

    return 'Unknown'


df_clean['GPU_Brand'] = df_clean['Graphics'].apply(extract_gpu_brand)
df_clean['GPU_Model'] = df_clean['Graphics'].apply(extract_gpu_model)

print("Distribusi GPU Brand:")
display(df_clean['GPU_Brand'].value_counts())

print("\nDistribusi GPU Model:")
display(df_clean['GPU_Model'].value_counts())

print("\nGPU yang masih Unknown:")
display(
    df_clean[df_clean['GPU_Model'] == 'Unknown'][
        ['Model', 'Graphics', 'GPU_Brand']
    ]
)



# -------------------- NOTEBOOK CELL 39 --------------------
# Koreksi salah penulisan nama GPU
df_clean.loc[22, 'GPU_Model'] = 'Radeon (Unspecified)'

# Koreksi variasi penulisan Intel Iris Xe
df_clean.loc[[313, 330], 'GPU_Model'] = 'Iris Xe'

# Informasi GPU tersedia pada kolom Model
df_clean.loc[65, 'GPU_Brand'] = 'NVIDIA'
df_clean.loc[65, 'GPU_Model'] = 'RTX 4090'

# Perbaiki brand dan model untuk NVIDIA yang hanya tertulis sebagian
df_clean.loc[212, 'GPU_Brand'] = 'NVIDIA'
df_clean.loc[212, 'GPU_Model'] = 'Unknown'

print("Distribusi GPU Brand:")
display(df_clean['GPU_Brand'].value_counts())

print("\nDistribusi GPU Model:")
display(df_clean['GPU_Model'].value_counts())

print("\nGPU Model Unknown:")
display(
    df_clean[df_clean['GPU_Model'] == 'Unknown'][
        ['Model', 'Graphics', 'GPU_Brand', 'GPU_Model']
    ]
)



# -------------------- NOTEBOOK CELL 40 --------------------
# Salin kolom Graphics
df_clean['Graphics_Clean'] = df_clean['Graphics']

# Tandai data yang bukan informasi GPU
invalid_graphics = [
    '256 GB SSD',
    '512 GB SSD',
    'Preinstalled Software: MS Office Home & Student 2021',
    '16 inches, 2880 x 1800 pixels, Touch Screen',
    '15.6 inches, 1920 x 1080 pixels',
    '18 inches, 3840 x 2400 pixels',
    '16 inches, 3200 x 2000 pixels',
    '16 inches, 1920 x 1080 pixels',
    'Linux OS',
    '1 Year Warranty'
]

for value in invalid_graphics:
    mask = (
        df_clean['Graphics_Clean']
        .str.replace('\u2009', ' ', regex=False)
        .str.replace('\u200e', '', regex=False)
        .str.strip()
        .str.lower()
        == value.lower()
    )

    df_clean.loc[mask, 'Graphics_Clean'] = np.nan

print("Jumlah Graphics kosong setelah pemeriksaan:",
      df_clean['Graphics_Clean'].isna().sum())

display(
    df_clean[df_clean['Graphics_Clean'].isna()][
        ['Model', 'Graphics', 'Graphics_Clean']
    ]
)



# -------------------- NOTEBOOK CELL 41 --------------------
# Pulihkan GPU dari kolom Model jika informasinya jelas
gpu_from_model = {
    65: 'NVIDIA GeForce RTX 4090'
}

for idx, gpu in gpu_from_model.items():
    df_clean.loc[idx, 'Graphics_Clean'] = gpu

# Tampilkan hasil pemeriksaan
display(
    df_clean.loc[
        [65, 84, 109, 111, 180, 327, 328, 341, 597, 598, 679, 772, 848],
        ['Model', 'Graphics', 'Graphics_Clean']
    ]
)

print(
    "\nJumlah Graphics_Clean kosong:",
    df_clean['Graphics_Clean'].isna().sum()
)



# -------------------- NOTEBOOK CELL 42 --------------------
df_clean['GPU_Brand'] = df_clean['Graphics_Clean'].apply(extract_gpu_brand)
df_clean['GPU_Model'] = df_clean['Graphics_Clean'].apply(extract_gpu_model)

df_clean['VRAM_GB'] = df_clean['Graphics_Clean'].apply(extract_vram)

print("Distribusi GPU Brand:")
display(df_clean['GPU_Brand'].value_counts())

print("\nDistribusi GPU Model:")
display(df_clean['GPU_Model'].value_counts())

print("\nStatistik VRAM:")
display(df_clean['VRAM_GB'].describe())

print("\nJumlah Graphics kosong:", df_clean['Graphics_Clean'].isna().sum())
print("Jumlah VRAM kosong:", df_clean['VRAM_GB'].isna().sum())



# -------------------- NOTEBOOK CELL 43 --------------------
def classify_gpu_type(row):
    brand = row['GPU_Brand']
    model = row['GPU_Model']

    if brand == 'Unknown' or model == 'Unknown':
        return 'Unknown'

    # NVIDIA umumnya dedicated
    if brand == 'NVIDIA':
        return 'Dedicated'

    # AMD GPU dedicated
    if model.startswith('RX '):
        return 'Dedicated'

    # Intel GPU dedicated
    if model.startswith('Arc'):
        return 'Dedicated'

    # GPU lainnya dikategorikan integrated
    if brand in ['Intel', 'AMD', 'Apple', 'ARM']:
        return 'Integrated'

    return 'Unknown'


df_clean['GPU_Type'] = df_clean.apply(classify_gpu_type, axis=1)

print("Distribusi GPU Type:")
display(df_clean['GPU_Type'].value_counts())

print("\nDistribusi GPU Type berdasarkan Brand:")
display(pd.crosstab(df_clean['GPU_Brand'], df_clean['GPU_Type']))



# -------------------- NOTEBOOK CELL 45 --------------------
# Ekstraksi ukuran layar
df_clean['Screen_Size'] = pd.to_numeric(
    df_clean['Display'].str.extract(r'([\d.]+)\s*inches', expand=False),
    errors='coerce'
)

# Ekstraksi resolusi
resolution = df_clean['Display'].str.extract(
    r'(\d+)\s*[x×]\s*(\d+)'
)

df_clean['Resolution_Width'] = pd.to_numeric(
    resolution[0], errors='coerce'
)

df_clean['Resolution_Height'] = pd.to_numeric(
    resolution[1], errors='coerce'
)

# Ekstraksi touchscreen
df_clean['Touchscreen'] = (
    df_clean['Display']
    .str.contains('Touch Screen', case=False, na=False)
    .astype(int)
)

print("Statistik ukuran layar:")
display(df_clean['Screen_Size'].describe())

print("\nStatistik resolusi:")
display(df_clean[['Resolution_Width', 'Resolution_Height']].describe())

print("\nDistribusi Touchscreen:")
display(df_clean['Touchscreen'].value_counts())

print("\nMissing values:")
display(df_clean[
    ['Screen_Size', 'Resolution_Width', 'Resolution_Height']
].isna().sum())



# -------------------- NOTEBOOK CELL 47 --------------------
# Pulihkan ukuran layar dari teks yang tersedia
df_clean.loc[109, 'Screen_Size'] = 15.6
df_clean.loc[709, 'Screen_Size'] = 16.0

# Tandai data Display yang bukan informasi layar
invalid_display_indices = [
    65, 84, 111, 180, 327, 328,
    341, 597, 598, 679, 772, 848
]

df_clean.loc[invalid_display_indices, 'Display'] = 'Unknown'

# Periksa hasil
display(
    df_clean.loc[
        [65, 84, 109, 111, 180, 327, 328, 341,
         597, 598, 679, 709, 772, 848],
        ['Model', 'Display', 'Screen_Size',
         'Resolution_Width', 'Resolution_Height']
    ]
)

print("\nMissing values:")
display(
    df_clean[
        ['Screen_Size', 'Resolution_Width', 'Resolution_Height']
    ].isna().sum()
)



# -------------------- NOTEBOOK CELL 48 --------------------
# Pulihkan informasi GPU dari kolom Display yang salah tempat
gpu_recovery = {
    84: '4 GB NVIDIA GeForce RTX 3050',
    327: '4 GB NVIDIA GeForce RTX 3050',
    597: '4 GB NVIDIA GeForce RTX 2050',
    598: '4 GB NVIDIA GeForce RTX 2050',
    772: '4 GB NVIDIA Quadro T600'
}

for idx, gpu in gpu_recovery.items():
    df_clean.loc[idx, 'Graphics_Clean'] = gpu

# Hitung ulang fitur GPU
df_clean['GPU_Brand'] = df_clean['Graphics_Clean'].apply(extract_gpu_brand)
df_clean['GPU_Model'] = df_clean['Graphics_Clean'].apply(extract_gpu_model)
df_clean['VRAM_GB'] = df_clean['Graphics_Clean'].apply(extract_vram)

print("Hasil pemulihan GPU:")
display(
    df_clean.loc[
        list(gpu_recovery.keys()),
        ['Model', 'Graphics_Clean', 'GPU_Brand', 'GPU_Model', 'VRAM_GB']
    ]
)



# -------------------- NOTEBOOK CELL 49 --------------------
df_clean.loc[772, 'GPU_Model'] = 'Quadro T600'

display(
    df_clean.loc[
        [84, 327, 597, 598, 772],
        ['GPU_Brand', 'GPU_Model', 'VRAM_GB']
    ]
)



# -------------------- NOTEBOOK CELL 50 --------------------
df_clean['GPU_Type'] = df_clean.apply(classify_gpu_type, axis=1)

display(df_clean['GPU_Type'].value_counts())



# -------------------- NOTEBOOK CELL 52 --------------------
os_invalid = [
    '1 Year Warranty',
    '2 Year Warranty',
    '15.6 inches, 1920 x 1080 pixels',
    'Backlit Keyboard',
    'Intel Iris Xe Graphics'
]

warranty_invalid = [
    'Backlit Keyboard',
    'Windows 11 OS',
    '1 USB 3.0 Ports',
    '2 USB 3.0 Ports',
    'Ports: 1 HDMI'
]

print("Data OS yang perlu diperiksa:")
display(
    df_clean.loc[
        df_clean['OS'].isin(os_invalid),
        ['Model', 'OS', 'Warranty', 'Display', 'Graphics_Clean']
    ]
)

print("\nData Warranty yang perlu diperiksa:")
display(
    df_clean.loc[
        df_clean['Warranty'].isin(warranty_invalid),
        ['Model', 'OS', 'Warranty', 'Display', 'Graphics_Clean']
    ]
)



# -------------------- NOTEBOOK CELL 53 --------------------
# Perbaikan OS berdasarkan informasi pada Model
os_corrections = {
    65: 'Windows 11 Home',
    109: 'Windows 11',
    111: 'Windows 11',
    180: 'Windows 11',
    213: 'DOS',
    217: 'DOS',
    328: 'Kirin OS',
    418: 'Prime OS',
    452: 'Prime OS',
    476: 'Windows 11 Home',
    550: 'Windows 11',
    848: 'Windows 11',
    914: 'Windows 11'
}

for idx, value in os_corrections.items():
    df_clean.loc[idx, 'OS'] = value

# Perbaikan Warranty yang informasinya jelas
warranty_corrections = {
    65: '2 Year Warranty',
    111: '1 Year Warranty',
    180: '1 Year Warranty',
    418: '1 Year Warranty',
    452: '1 Year Warranty',
    848: '1 Year Warranty'
}

for idx, value in warranty_corrections.items():
    df_clean.loc[idx, 'Warranty'] = value

# Rapikan spasi pada OS
df_clean['OS'] = df_clean['OS'].str.replace(r'\s+', ' ', regex=True).str.strip()

# Ganti nilai yang bukan informasi OS
invalid_os_indices = [67, 84, 327, 597, 598, 679, 772]
df_clean.loc[invalid_os_indices, 'OS'] = 'Unknown'

# Ganti nilai yang bukan informasi Warranty
invalid_warranty_indices = [
    67, 84, 109, 111, 180, 212, 213, 217,
    327, 344, 418, 452, 476, 550, 597, 598,
    772, 848, 914
]
df_clean.loc[invalid_warranty_indices, 'Warranty'] = 'Unknown'

print("Distribusi OS:")
display(df_clean['OS'].value_counts(dropna=False))

print("\nDistribusi Warranty:")
display(df_clean['Warranty'].value_counts(dropna=False))



# -------------------- NOTEBOOK CELL 55 --------------------
df_clean.loc[676, 'OS'] = 'JioOS'
df_clean.loc[676, 'Warranty'] = '1 Year Warranty'

display(
    df_clean.loc[
        [676],
        ['Model', 'OS', 'Warranty']
    ]
)



# -------------------- NOTEBOOK CELL 56 --------------------
df_clean['OS'] = df_clean['OS'].replace({
    'Windows 11 OS': 'Windows 11',
    'Windows 11  OS': 'Windows 11',
    'Windows 11 Home': 'Windows 11',
    'Windows 11 Pro OS': 'Windows 11',
    'Windows 10 OS': 'Windows 10',
    'Windows 10  OS': 'Windows 10',
    'Windows OS': 'Windows',
    'Mac OS': 'macOS',
    'DOS OS': 'DOS',
    'DOS 3.0 OS': 'DOS',
    'Ubuntu OS': 'Ubuntu',
    'Chrome OS': 'ChromeOS'
})

display(df_clean['OS'].value_counts(dropna=False))



# -------------------- NOTEBOOK CELL 58 --------------------
# Kolom yang akan digunakan untuk pemodelan
selected_columns = [
    'Price',
    'RAM_GB',
    'RAM_Type',
    'Storage_GB',
    'CPU_Brand',
    'CPU_Class',
    'CPU_Cores',
    'CPU_Threads',
    'GPU_Brand',
    'GPU_Model',
    'GPU_Type',
    'VRAM_GB',
    'Screen_Size',
    'Resolution_Width',
    'Resolution_Height',
    'Touchscreen',
    'OS',
    'Warranty'
]

df_model = df_clean[selected_columns].copy()

print("Ukuran dataset untuk pemodelan:", df_model.shape)
display(df_model.head())



# -------------------- NOTEBOOK CELL 63 --------------------
# Perbaikan berdasarkan informasi pada Model
df_model.loc[180, 'RAM_GB'] = 16

# RAM indeks 679 tidak diketahui secara pasti
df_model.loc[679, 'RAM_GB'] = np.nan

# Storage indeks 679: 1 TB = 1024 GB
df_model.loc[679, 'Storage_GB'] = 1024

# Periksa hasil perbaikan
display(
    df_model.loc[
        [180, 679],
        ['RAM_GB', 'Storage_GB']
    ]
)

print("\nStatistik RAM dan Storage setelah perbaikan:")
display(df_model[['RAM_GB', 'Storage_GB']].describe())



# -------------------- NOTEBOOK CELL 65 --------------------
# Membuat fitur rasio aspek layar
df_model['Aspect_Ratio'] = (
    df_model['Resolution_Width'] /
    df_model['Resolution_Height']
)

print("Statistik Aspect Ratio:")
display(df_model['Aspect_Ratio'].describe())

print("\nContoh data:")
display(
    df_model[
        ['Resolution_Width', 'Resolution_Height', 'Aspect_Ratio']
    ].head(10)
)



# -------------------- NOTEBOOK CELL 66 --------------------
# Menghapus fitur resolusi asli karena sudah direpresentasikan oleh Aspect_Ratio
df_model = df_model.drop(
    columns=['Resolution_Width', 'Resolution_Height']
)

print("Ukuran dataset:", df_model.shape)
print("\nDaftar fitur:")
print(df_model.columns.tolist())

print("\nMissing values:")
display(
    df_model.isnull().sum().to_frame('Missing Values')
    .query('`Missing Values` > 0')
)



# Sinkronkan koreksi final kapasitas RAM/storage ke dataset hasil cleaning.
df_clean.loc[180, "RAM_GB"] = 16
df_clean.loc[679, "RAM_GB"] = np.nan
df_clean.loc[679, "Storage_GB"] = 1024

# -------------------- AUDIT LOG KOREKSI MANUAL --------------------
# Daftar berikut mencatat koreksi berbasis pemeriksaan judul/model yang ada di notebook.
manual_fixes = [
    (109, "RAM_GB", "missing/keliru", 8, "Kapasitas RAM diverifikasi dari informasi model"),
    (328, "RAM_GB", "missing/keliru", 16, "Kapasitas RAM diverifikasi dari informasi model"),
    (341, "RAM_GB", "missing/keliru", 4, "Kapasitas RAM diverifikasi dari informasi model"),
    (341, "CPU_Cores", "missing", 8, "Octa Core pada spesifikasi"),
    (180, "RAM_GB", "512 (hasil ekstraksi keliru)", 16, "Koreksi berdasarkan judul/model"),
    (679, "RAM_GB", "1 (tidak dapat dipastikan)", "NaN", "Dibiarkan missing karena tipe/kapasitas RAM tidak jelas"),
    (679, "Storage_GB", "1 TB", 1024, "Konversi 1 TB menjadi 1024 GB"),
    (65, "GPU_Model", "missing/keliru", "RTX 4090", "Informasi GPU diverifikasi dari judul/model"),
    (84, "Graphics_Clean", "informasi GPU bergeser", "4 GB NVIDIA GeForce RTX 3050", "Pemulihan informasi GPU dari kolom Display"),
    (327, "Graphics_Clean", "informasi GPU bergeser", "4 GB NVIDIA GeForce RTX 3050", "Pemulihan informasi GPU dari kolom Display"),
    (597, "Graphics_Clean", "informasi GPU bergeser", "4 GB NVIDIA GeForce RTX 2050", "Pemulihan informasi GPU dari kolom Display"),
    (598, "Graphics_Clean", "informasi GPU bergeser", "4 GB NVIDIA GeForce RTX 2050", "Pemulihan informasi GPU dari kolom Display"),
    (772, "Graphics_Clean", "informasi GPU bergeser", "4 GB NVIDIA Quadro T600", "Pemulihan informasi GPU dari kolom Display"),
]
for idx, col, old, new, reason in manual_fixes:
    log_fix(idx, col, old, new, reason)

# Catat perubahan kategori OS/Warranty yang dilakukan pada notebook.
for idx, val in {65:'Windows 11 Home',109:'Windows 11',111:'Windows 11',180:'Windows 11',213:'DOS',217:'DOS',328:'Kirin OS',418:'Prime OS',452:'Prime OS',476:'Windows 11 Home',550:'Windows 11',848:'Windows 11',914:'Windows 11',676:'JioOS'}.items():
    log_fix(idx, 'OS', 'nilai salah tempat/tidak konsisten', val, 'Koreksi berdasarkan pemeriksaan spesifikasi/model')

# Dataset hasil cleaning mencakup kolom asli + hasil feature engineering.
df_clean.to_csv(OUT_DIR / "laptop_clean.csv", index=False)
df_model.to_csv(OUT_DIR / "laptop_features.csv", index=False)
pd.DataFrame(cleaning_log).to_csv(OUT_DIR / "cleaning_log.csv", index=False)

print("\n[6/6] Train-test split dan preprocessing")
X = df_model.drop(columns=["Price"])
y = df_model["Price"]
numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

# Pertahankan skema split dari notebook: 80:20, random_state=42.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def build_preprocessor(scale_numeric=True):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(steps=numeric_steps)
    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer(transformers=[
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features)
    ])

# Fit hanya pada data training untuk mencegah data leakage.
preprocessor_linear = build_preprocessor(scale_numeric=True)
preprocessor_tree = build_preprocessor(scale_numeric=False)
X_train_linear = preprocessor_linear.fit_transform(X_train)
X_test_linear = preprocessor_linear.transform(X_test)
X_train_tree = preprocessor_tree.fit_transform(X_train)
X_test_tree = preprocessor_tree.transform(X_test)

def to_dense_array(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)

# Simpan split mentah agar bisa dipakai ulang untuk pipeline modeling/CV.
train = X_train.copy()
train["Price"] = y_train
test = X_test.copy()
test["Price"] = y_test
train.to_csv(OUT_DIR / "train.csv", index=False)
test.to_csv(OUT_DIR / "test.csv", index=False)

# Simpan matriks hasil preprocessing untuk inspeksi/eksperimen langsung.
feature_names_linear = preprocessor_linear.get_feature_names_out().tolist()
feature_names_tree = preprocessor_tree.get_feature_names_out().tolist()
pd.DataFrame(to_dense_array(X_train_linear), columns=feature_names_linear, index=X_train.index).to_csv(OUT_DIR / "X_train_linear.csv", index=False)
pd.DataFrame(to_dense_array(X_test_linear), columns=feature_names_linear, index=X_test.index).to_csv(OUT_DIR / "X_test_linear.csv", index=False)
pd.DataFrame(to_dense_array(X_train_tree), columns=feature_names_tree, index=X_train.index).to_csv(OUT_DIR / "X_train_tree.csv", index=False)
pd.DataFrame(to_dense_array(X_test_tree), columns=feature_names_tree, index=X_test.index).to_csv(OUT_DIR / "X_test_tree.csv", index=False)
y_train.to_frame("Price").to_csv(OUT_DIR / "y_train.csv", index=False)
y_test.to_frame("Price").to_csv(OUT_DIR / "y_test.csv", index=False)

joblib.dump(preprocessor_linear, OUT_DIR / "preprocessor_linear.joblib")
joblib.dump(preprocessor_tree, OUT_DIR / "preprocessor_tree.joblib")
with open(OUT_DIR / "feature_lists.json", "w", encoding="utf-8") as f:
    json.dump({
        "target": "Price",
        "target_unit": "INR",
        "numeric": numeric_features,
        "categorical": categorical_features,
        "features_before_encoding": X.columns.tolist(),
        "features_after_encoding_linear": feature_names_linear,
        "features_after_encoding_tree": feature_names_tree,
        "split": {"test_size": 0.2, "random_state": 42}
    }, f, indent=2, ensure_ascii=False)

# Ringkasan validasi hasil.
assert len(df_clean) == len(df_model), "Jumlah baris berubah antara cleaning dan feature set."
assert X_train.shape[0] + X_test.shape[0] == len(df_model), "Jumlah data split tidak sesuai."
assert not np.isnan(to_dense_array(X_train_linear)).any(), "Masih ada NaN pada train setelah preprocessing."
assert not np.isnan(to_dense_array(X_test_linear)).any(), "Masih ada NaN pada test setelah preprocessing."

print("\n================ RINGKASAN AKHIR ================")
print(f"Dataset awal             : {df.shape}")
print(f"Dataset cleaned          : {df_clean.shape}")
print(f"Dataset fitur modeling   : {df_model.shape}")
print(f"Fitur numerik/kategorikal: {len(numeric_features)} / {len(categorical_features)}")
print(f"Train / test             : {X_train.shape} / {X_test.shape}")
print(f"Linear transformed       : {to_dense_array(X_train_linear).shape} / {to_dense_array(X_test_linear).shape}")
print(f"Tree transformed         : {to_dense_array(X_train_tree).shape} / {to_dense_array(X_test_tree).shape}")
print(f"Cleaning log             : {len(cleaning_log)} catatan")
print(f"Semua output tersimpan di: {OUT_DIR.resolve()}")
print("==================================================")
