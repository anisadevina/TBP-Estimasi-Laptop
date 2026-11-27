# app/app.py
# Dashboard Estimasi Harga Laptop (Streamlit)
#
# Halaman:
#   1. Prediksi          → form spesifikasi → estimasi harga + rentang + laptop mirip + what-if
#   2. Eksplorasi Data   → distribusi harga, boxplot, korelasi
#   3. Perbandingan Model→ tabel/grafik metrik, aktual vs prediksi
#   4. Faktor Harga      → feature importance
#   5. Keandalan Model   → error per segmen harga, overfitting, keterbatasan
#
# Jalankan dari folder utama proyek:
#   streamlit run app/app.py

from __future__ import annotations

import warnings
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

ROOT        = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results" / "modeling"
DATA_CANDIDATES = [ROOT / "data" / "processed" / "laptop_features.csv",
                   ROOT / "app" / "data" / "laptop_features.csv"]

MODEL_FILES = {
    "Random Forest":              "random_forest.joblib",
    "Multiple Linear Regression": "multiple_linear_regression.joblib",
    "Decision Tree":              "decision_tree.joblib",
}
NUMERIC_COLS = ["RAM_GB", "Storage_GB", "CPU_Cores", "CPU_Threads", "VRAM_GB",
                "Screen_Size", "Touchscreen", "Aspect_Ratio"]
CAT_COLS     = ["RAM_Type", "CPU_Brand", "CPU_Class", "GPU_Brand", "GPU_Model",
                "GPU_Type", "OS", "Warranty"]

def _st_version() -> tuple:
    try:
        return tuple(int(x) for x in st.__version__.split(".")[:2])
    except Exception:
        return (0, 0)


# Streamlit >= 1.50 memakai width="stretch"; versi lama memakai use_container_width
STRETCH = {"width": "stretch"} if _st_version() >= (1, 50) else {"use_container_width": True}

st.set_page_config(page_title="Estimasi Harga Laptop", page_icon="💻", layout="wide")


# ─────────────────────────────────────────────────────────────
# Helper format
# ─────────────────────────────────────────────────────────────
def _fmt(x: float, sym: str) -> str:
    sign = "-" if x < 0 else ""
    return sign + sym + f"{abs(x):,.0f}".replace(",", ".")


def inr(x: float) -> str:
    """Format nilai asli (INR), tanpa konversi."""
    return _fmt(x, "₹")


# Mata uang tampilan; diatur lewat sidebar (nilai awal = INR, tanpa konversi)
CUR, RATE = "INR", 1.0


def money(x: float) -> str:
    """Format nilai harga (dalam INR) ke mata uang tampilan yang dipilih."""
    return _fmt(x * RATE, "Rp" if CUR == "IDR" else "₹")


# ─────────────────────────────────────────────────────────────
# Loader (di-cache)
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_dataset() -> pd.DataFrame:
    for p in DATA_CANDIDATES:
        if p.exists():
            return pd.read_csv(p)
    st.error("Dataset tidak ditemukan. Jalankan `python src/preprocessing.py` "
             "atau pastikan `app/data/laptop_features.csv` ada.")
    st.stop()


@st.cache_data
def load_csv(name: str, **kw) -> pd.DataFrame | None:
    p = RESULTS_DIR / name
    return pd.read_csv(p, **kw) if p.exists() else None


def _model_files() -> list:
    return [RESULTS_DIR / "best_model.joblib"] + \
           [RESULTS_DIR / "models" / f for f in MODEL_FILES.values()]


def _models_signature() -> tuple:
    """Tanda waktu file model: bila model dilatih ulang, cache otomatis diperbarui."""
    return tuple((p.name, p.stat().st_mtime if p.exists() else 0) for p in _model_files())


@st.cache_resource
def _load_models_cached(signature: tuple) -> dict:
    out, errors = {}, []
    for name, fname in MODEL_FILES.items():
        p = RESULTS_DIR / "models" / fname
        if p.exists():
            try:
                out[name] = joblib.load(p)
            except Exception as e:  # biasanya beda versi numpy / scikit-learn
                errors.append(f"{fname}: {type(e).__name__}: {e}")
    best = RESULTS_DIR / "best_model.joblib"
    if best.exists():
        try:
            out["__best__"] = joblib.load(best)
        except Exception as e:
            errors.append(f"best_model.joblib: {type(e).__name__}: {e}")
    out["__errors__"] = errors
    return out


def load_models() -> dict:
    return _load_models_cached(_models_signature())


def env_info() -> str:
    """Info lingkungan untuk diagnosis masalah versi."""
    import datetime
    import sys
    import sklearn
    lines = [f"Python   : {sys.version.split()[0]}  ({sys.executable})",
             f"numpy    : {np.__version__}",
             f"pandas   : {pd.__version__}",
             f"sklearn  : {sklearn.__version__}",
             f"streamlit: {st.__version__}", "", "Waktu simpan file model:"]
    for p in _model_files():
        t = (datetime.datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
             if p.exists() else "TIDAK ADA")
        lines.append(f"  {p.name:<34} {t}")
    return "\n".join(lines)


df      = load_dataset()
models  = load_models()
comp_df = load_csv("model_comparison.csv")
seg_df  = load_csv("error_per_segmen.csv")
imp_df  = load_csv("feature_importance.csv", index_col=0)
pred_df = load_csv("prediksi_test.csv")
coef_df = load_csv("koefisien_mlr.csv")

_RETRAIN_HINT = ("Jalankan ulang training di komputer ini agar model sesuai versi "
                 "scikit-learn yang terpasang:\n\n"
                 "`python src/preprocessing.py` lalu `python src/modeling.py`, "
                 "kemudian `python src/evaluation.py`.")

if "__best__" not in models:
    st.error("Model belum bisa dimuat. " + _RETRAIN_HINT)
    if models.get("__errors__"):
        st.code("\n".join(models["__errors__"]))
    st.markdown("**Info lingkungan** (bandingkan dengan Python yang dipakai saat training; "
                "waktu simpan model harus sesudah Anda menjalankan `modeling.py`):")
    st.code(env_info())
    st.stop()

try:  # uji cepat: model yang berhasil dimuat belum tentu bisa dipakai di versi lain
    models["__best__"].predict(df[NUMERIC_COLS + CAT_COLS].head(1))
except Exception as e:
    st.error("Model berhasil dimuat tetapi gagal dipakai (kemungkinan beda versi "
             "scikit-learn). " + _RETRAIN_HINT)
    st.code(f"{type(e).__name__}: {e}")
    st.code(env_info())
    st.stop()

BEST_NAME = "Random Forest"
if comp_df is not None and (comp_df["Terbaik"] == "★").any():
    BEST_NAME = comp_df.loc[comp_df["Terbaik"] == "★", "Model"].iloc[0]


# ─────────────────────────────────────────────────────────────
# Fungsi bantu prediksi
# ─────────────────────────────────────────────────────────────
def mode_of(s: pd.Series, default=None):
    s = s.dropna()
    return s.mode().iloc[0] if len(s) else default


def predict_rows(rows: list[dict], model_key: str = "__best__") -> np.ndarray:
    X = pd.DataFrame(rows)[NUMERIC_COLS + CAT_COLS]
    return np.clip(models[model_key].predict(X), 0, None)


def segment_of(price: float) -> int:
    """Segmen 0-3 berdasar kuartil harga aktual pada data test."""
    if pred_df is None:
        return 0
    q = pred_df["Harga_Aktual"].quantile([.25, .5, .75]).values
    return int(np.searchsorted(q, price))


# ─────────────────────────────────────────────────────────────
# Sidebar: pilihan mata uang tampilan
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Pengaturan tampilan")
    _choice = st.radio("Mata uang harga", ["INR (₹)", "IDR (Rp)"],
                       help="Data dan model memakai INR. Pilihan IDR hanya mengonversi "
                            "tampilan dengan kurs di bawah.")
    CUR = "IDR" if _choice.startswith("IDR") else "INR"
    if CUR == "IDR":
        RATE = float(st.number_input("Kurs: 1 INR = Rp", 50.0, 500.0, 185.0, step=1.0,
                                     help="Nilai awal hanya perkiraan kasar kurs tengah. "
                                          "Ubah sesuai kurs terkini."))
        st.caption("Konversi linear sederhana: nilai INR × kurs. Bukan harga jual di "
                   "Indonesia (belum termasuk pajak, bea masuk, atau margin).")
    else:
        RATE = 1.0

# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────
st.title("💻 Estimasi Harga Laptop")
_cur_note = ("Harga dalam Indian Rupee (INR)" if CUR == "INR" else
             f"Harga dalam Rupiah (IDR), dikonversi dari INR dengan kurs 1 INR = Rp{RATE:g}")
st.caption(f"Model aktif: **{BEST_NAME}** (dipilih berdasarkan CV RMSE terendah) · "
           f"Dataset: {len(df)} laptop · {_cur_note}")

MONEY_METRICS = ["CV_RMSE", "CV_RMSE_std", "Test_MAE", "Test_RMSE"]   # skala = kurs
SQUARED_METRICS = ["Test_MSE"]                                       # skala = kurs²


def scale_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for c in MONEY_METRICS:
        if c in out.columns:
            out[c] = out[c] * RATE
    for c in SQUARED_METRICS:
        if c in out.columns:
            out[c] = out[c] * RATE ** 2
    return out


tab_pred, tab_eda, tab_model, tab_feat, tab_rel = st.tabs(
    ["🔮 Prediksi", "📊 Eksplorasi Data", "🏁 Perbandingan Model",
     "🔑 Faktor Harga", "🛡️ Keandalan Model"])


# ═════════════════════════════════════════════════════════════
# TAB 1 — PREDIKSI
# ═════════════════════════════════════════════════════════════
with tab_pred:
    left, right = st.columns([1.1, 1])

    with left:
        st.subheader("Spesifikasi laptop")
        c1, c2 = st.columns(2)

        ram_opts = sorted(df["RAM_GB"].dropna().unique())
        ram = c1.selectbox("RAM (GB)", ram_opts,
                           index=ram_opts.index(16.0) if 16.0 in ram_opts else 0,
                           format_func=lambda v: f"{v:g}")
        sto_opts = sorted(df["Storage_GB"].dropna().unique())
        storage = c2.selectbox("Penyimpanan (GB)", sto_opts,
                               index=sto_opts.index(512.0) if 512.0 in sto_opts else 0,
                               format_func=lambda v: f"{v:g}")

        rt_opts = df["RAM_Type"].value_counts().index.tolist()
        rt_def = mode_of(df.loc[df["RAM_GB"] == ram, "RAM_Type"], rt_opts[0])
        ram_type = c1.selectbox("Tipe RAM", rt_opts, index=rt_opts.index(rt_def),
                                key=f"ramtype_{ram}")

        scr_opts = sorted(df["Screen_Size"].dropna().unique())
        screen = c2.selectbox("Ukuran layar (inci)", scr_opts,
                              index=scr_opts.index(15.6) if 15.6 in scr_opts else 0)

        cpu_opts = df["CPU_Class"].value_counts().index.tolist()
        cpu_brand_map = df.groupby("CPU_Class")["CPU_Brand"].agg(mode_of).to_dict()
        cpu = c1.selectbox("Prosesor", cpu_opts,
                           index=cpu_opts.index("Core i5") if "Core i5" in cpu_opts else 0,
                           format_func=lambda v: f"{v} ({cpu_brand_map[v]})")

        gpu_opts = df["GPU_Model"].value_counts().index.tolist()
        gpu_brand_map = df.groupby("GPU_Model")["GPU_Brand"].agg(mode_of).to_dict()
        gpu_type_map  = df.groupby("GPU_Model")["GPU_Type"].agg(mode_of).to_dict()
        gpu = c2.selectbox("Kartu grafis", gpu_opts,
                           index=gpu_opts.index("Iris Xe") if "Iris Xe" in gpu_opts else 0,
                           format_func=lambda v: f"{v} ({gpu_brand_map[v]})")

        os_opts = df["OS"].value_counts().index.tolist()
        os_ = c1.selectbox("Sistem operasi", os_opts,
                           index=os_opts.index("Windows 11") if "Windows 11" in os_opts else 0)
        wr_opts = df["Warranty"].value_counts().index.tolist()
        warranty = c2.selectbox("Garansi", wr_opts, index=0)
        touch = st.checkbox("Layar sentuh (touchscreen)")

        # Nilai default otomatis dari data, bisa diubah di bagian lanjutan
        cores_def   = int(df.loc[df["CPU_Class"] == cpu, "CPU_Cores"].median())
        threads_def = int(df.loc[df["CPU_Class"] == cpu, "CPU_Threads"].median())
        vram_med    = df.loc[df["GPU_Model"] == gpu, "VRAM_GB"].median()
        aspect_def  = float(mode_of(df.loc[df["Screen_Size"] == screen, "Aspect_Ratio"], 1.7778))

        with st.expander("Pengaturan lanjutan (terisi otomatis dari data)"):
            a1, a2, a3, a4 = st.columns(4)
            cores = a1.number_input("Core CPU", 1, 32, cores_def, key=f"cores_{cpu}")
            threads = a2.number_input("Thread CPU", 1, 64, threads_def, key=f"thr_{cpu}")
            vram_opts = ["Otomatis / tidak ada"] + [2, 4, 6, 8, 12, 16]
            vram_def = (int(vram_med) if not np.isnan(vram_med) and int(vram_med) in vram_opts
                        else "Otomatis / tidak ada")
            vram_sel = a3.selectbox("VRAM (GB)", vram_opts, index=vram_opts.index(vram_def),
                                    key=f"vram_{gpu}")
            aspect = a4.number_input("Aspect ratio", 1.0, 2.0, round(aspect_def, 4),
                                     step=0.0001, format="%.4f", key=f"asp_{screen}")
            st.caption(f"Otomatis: CPU {cpu_brand_map[cpu]} · GPU {gpu_brand_map[gpu]} "
                       f"({gpu_type_map[gpu]})")

    vram = np.nan if vram_sel == "Otomatis / tidak ada" else float(vram_sel)
    spec = {"RAM_GB": ram, "Storage_GB": storage, "CPU_Cores": float(cores),
            "CPU_Threads": float(threads), "VRAM_GB": vram, "Screen_Size": screen,
            "Touchscreen": int(touch), "Aspect_Ratio": aspect, "RAM_Type": ram_type,
            "CPU_Brand": cpu_brand_map[cpu], "CPU_Class": cpu,
            "GPU_Brand": gpu_brand_map[gpu], "GPU_Model": gpu,
            "GPU_Type": gpu_type_map[gpu], "OS": os_, "Warranty": warranty}

    price = float(predict_rows([spec])[0])
    seg = segment_of(price)
    seg_mae = float(seg_df["MAE"].iloc[seg]) if seg_df is not None else np.nan
    seg_name = seg_df["segmen"].iloc[seg] if seg_df is not None else "-"

    with right:
        st.subheader("Hasil estimasi")
        st.metric(f"Estimasi harga ({BEST_NAME})", money(price))
        if not np.isnan(seg_mae):
            lo, hi = max(price - seg_mae, 0), price + seg_mae
            st.write(f"Rentang perkiraan kasar: **{money(lo)} – {money(hi)}**")
            st.caption(f"Rentang = estimasi ± rata-rata galat model (MAE) pada segmen "
                       f"**{seg_name}** di data uji ({money(seg_mae)}). "
                       "Ini panduan kasar, bukan interval kepercayaan statistik.")

        if CUR == "IDR":
            st.caption(f"Nilai asli model: {inr(price)} (INR), dikonversi dengan kurs "
                       f"1 INR = Rp{RATE:g}. Konversi kasar, bukan harga jual di Indonesia.")
        else:
            st.caption("Ingin melihat dalam rupiah? Ganti mata uang di sidebar kiri.")

        if seg == 3:
            st.warning("Estimasi masuk segmen termahal. Galat model di segmen ini jauh "
                       "lebih besar daripada segmen lain, jadi anggap hasilnya sangat kasar.")

        # Peringatan konsistensi spesifikasi
        if threads < cores:
            st.warning("Jumlah thread lebih kecil dari jumlah core. Periksa kembali input.")
        n_combo = int(((df["CPU_Class"] == cpu) & (df["GPU_Model"] == gpu)).sum())
        if n_combo == 0:
            st.info("Kombinasi prosesor dan kartu grafis ini tidak ada di data latih, "
                    "sehingga estimasi kurang andal.")
        elif n_combo < 3:
            st.info(f"Kombinasi prosesor dan kartu grafis ini hanya muncul {n_combo}× di data.")

        # Perbandingan 3 model
        st.markdown("**Prediksi ketiga model**")
        rows = []
        for name in MODEL_FILES:
            if name in models:
                rows.append({"Model": name + (" ★" if name == BEST_NAME else ""),
                             "Estimasi": money(float(predict_rows([spec], name)[0]))})
        st.dataframe(pd.DataFrame(rows), hide_index=True, **STRETCH)

    st.divider()
    w1, w2 = st.columns(2)

    # Laptop mirip
    with w1:
        st.subheader("Laptop mirip di dataset")
        d = df.copy()
        score = (np.abs(np.log2(d["RAM_GB"]) - np.log2(ram))
                 + np.abs(np.log2(d["Storage_GB"]) - np.log2(storage))
                 + (d["CPU_Class"] != cpu) * 1.0
                 + (d["GPU_Model"] != gpu) * 1.0
                 + np.abs(d["Screen_Size"] - screen) / 2
                 + (d["OS"] != os_) * 0.3)
        sim = d.assign(_s=score).nsmallest(3, "_s")
        show = sim[["Price", "RAM_GB", "Storage_GB", "CPU_Class", "GPU_Model",
                    "Screen_Size", "OS"]].copy()
        show["Price"] = show["Price"].map(money)
        show.columns = ["Harga", "RAM", "SSD/HDD", "CPU", "GPU", "Layar", "OS"]
        st.dataframe(show, hide_index=True, **STRETCH)
        st.caption("Tiga baris dengan spesifikasi paling dekat, untuk membandingkan "
                   "estimasi dengan harga nyata di data.")

    # What-if
    with w2:
        st.subheader("Simulasi what-if")
        var = st.radio("Ubah spesifikasi", ["RAM_GB", "Storage_GB"], horizontal=True,
                       format_func=lambda v: "RAM" if v == "RAM_GB" else "Penyimpanan")
        opts = sorted(df[var].dropna().unique())
        sims = []
        for v in opts:
            r = dict(spec); r[var] = v
            sims.append(r)
        sim_prices = predict_rows(sims)
        wi = pd.DataFrame({"Nilai (GB)": opts, "Estimasi": sim_prices * RATE})
        base = alt.Chart(wi).encode(
            x=alt.X("Nilai (GB):O", title=f"{'RAM' if var == 'RAM_GB' else 'Penyimpanan'} (GB)"),
            y=alt.Y("Estimasi:Q", title=f"Estimasi harga ({CUR})"))
        cur = alt.Chart(pd.DataFrame({"Nilai (GB)": [spec[var]]})).mark_rule(
            strokeDash=[4, 4], color="#d62728").encode(x="Nilai (GB):O")
        st.altair_chart(base.mark_line(point=True) + cur, **STRETCH)
        st.caption("Garis merah = nilai yang sedang dipilih. Spesifikasi lain tetap. "
                   "Ini menunjukkan sensitivitas model, bukan bukti sebab-akibat.")

    st.caption("Estimasi berasal dari model machine learning yang dilatih pada data "
               "harga laptop (data asli dalam INR). Hasil bukan penawaran atau harga pasti.")


# ═════════════════════════════════════════════════════════════
# TAB 2 — EKSPLORASI DATA
# ═════════════════════════════════════════════════════════════
with tab_eda:
    st.subheader("Gambaran dataset")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Jumlah laptop", f"{len(df)}")
    m2.metric("Harga median", money(df["Price"].median()))
    m3.metric("Harga terendah", money(df["Price"].min()))
    m4.metric("Harga tertinggi", money(df["Price"].max()))

    st.markdown("**Distribusi harga**")
    hist = alt.Chart(df.assign(Price=df["Price"] * RATE)).mark_bar().encode(
        x=alt.X("Price:Q", bin=alt.Bin(maxbins=40), title=f"Harga ({CUR})"),
        y=alt.Y("count()", title="Jumlah laptop"))
    st.altair_chart(hist, **STRETCH)
    st.caption("Distribusi menceng ke kanan: sebagian besar laptop murah-menengah, "
               "dengan ekor panjang laptop premium. Ekor inilah yang paling sulit diprediksi.")

    st.markdown("**Harga menurut spesifikasi**")
    var_map = {"RAM_GB": "RAM (GB)", "Storage_GB": "Penyimpanan (GB)",
               "CPU_Class": "Prosesor", "GPU_Brand": "Merek GPU",
               "GPU_Type": "Tipe GPU", "OS": "Sistem operasi",
               "Screen_Size": "Ukuran layar"}
    var = st.selectbox("Kelompokkan berdasarkan", list(var_map),
                       format_func=lambda v: var_map[v])
    d = df.assign(Price=df["Price"] * RATE)
    if d[var].dtype == object or str(d[var].dtype).startswith("str"):
        keep = d[var].value_counts()
        d = d[d[var].isin(keep[keep >= 10].index)]
        st.caption("Kategori dengan kurang dari 10 laptop tidak ditampilkan.")
    box = alt.Chart(d).mark_boxplot(extent="min-max").encode(
        x=alt.X(f"{var}:N", title=var_map[var], sort=None),
        y=alt.Y("Price:Q", title=f"Harga ({CUR})"))
    st.altair_chart(box, **STRETCH)

    st.markdown("**Korelasi antar fitur numerik**")
    corr = df[NUMERIC_COLS + ["Price"]].corr(numeric_only=True)
    cl = corr.reset_index().melt("index")
    cl.columns = ["a", "b", "r"]
    heat = alt.Chart(cl).mark_rect().encode(
        x=alt.X("a:N", title=None), y=alt.Y("b:N", title=None),
        color=alt.Color("r:Q", scale=alt.Scale(domain=[-1, 1], scheme="redblue"),
                        title="Korelasi"))
    txt = alt.Chart(cl).mark_text(size=11).encode(
        x="a:N", y="b:N", text=alt.Text("r:Q", format=".2f"),
        color=alt.condition("abs(datum.r) > 0.6", alt.value("white"), alt.value("black")))
    st.altair_chart(heat + txt, **STRETCH)
    st.caption("Fitur seperti RAM, jumlah core, dan thread saling berkorelasi (multikolinearitas), "
               "yang memengaruhi interpretasi koefisien regresi linear.")

    with st.expander("Lihat data"):
        st.caption(f"Kolom Price ditampilkan dalam {CUR}.")
        st.dataframe(df.assign(Price=df["Price"] * RATE), **STRETCH)


# ═════════════════════════════════════════════════════════════
# TAB 3 — PERBANDINGAN MODEL
# ═════════════════════════════════════════════════════════════
with tab_model:
    st.subheader("Perbandingan tiga model")
    if comp_df is None:
        st.info("`model_comparison.csv` belum ada. Jalankan `python src/evaluation.py`.")
    else:
        st.info(f"**Model terpilih: {BEST_NAME}**, berdasarkan **CV RMSE** terendah (5-fold pada "
                "data latih). Data uji hanya dipakai untuk melaporkan performa akhir, "
                "bukan untuk memilih model.")
        comp_money = scale_metrics(comp_df)
        view = comp_money.rename(columns={"Test_MAPE_%": "Test_MAPE (%)"})
        st.dataframe(view, hide_index=True, **STRETCH)
        st.caption(f"Metrik galat (RMSE, MAE, MSE) dalam {CUR}"
                   + (f"; MSE dalam {CUR}². Nilai dikonversi dari INR dengan kurs "
                      f"1 INR = Rp{RATE:g}. R² dan MAPE tidak berubah." if CUR == "IDR" else "."))

        metric_opts = {"CV_RMSE": "CV RMSE", "Test_RMSE": "Test RMSE",
                       "Test_MAE": "Test MAE", "Test_R2": "Test R²",
                       "Test_MAPE_%": "Test MAPE (%)"}
        metric = st.selectbox("Metrik", list(metric_opts), format_func=lambda v: metric_opts[v])
        better = "lebih tinggi lebih baik" if metric == "Test_R2" else "lebih rendah lebih baik"
        unit = f" {CUR}," if metric in MONEY_METRICS else ""
        bar = alt.Chart(comp_money).mark_bar().encode(
            x=alt.X("Model:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y(f"{metric}:Q", title=f"{metric_opts[metric]} ({unit} {better})".replace("( ", "(")),
            color=alt.Color("Model:N", legend=None),
            tooltip=["Model", alt.Tooltip(f"{metric}:Q", format=",.2f")])
        st.altair_chart(bar, **STRETCH)

        win = load_csv("pemenang_per_metrik.csv")
        if win is not None:
            st.markdown("**Pemenang tiap metrik**")
            win = win.copy()
            win["Nilai"] = [v * RATE if m in MONEY_METRICS else v
                            for m, v in zip(win["Metrik"], win["Nilai"])]
            st.dataframe(win, hide_index=True, **STRETCH)
        st.caption("Metrik bisa memberi pemenang berbeda. RMSE dan R² sangat dipengaruhi "
                   "oleh beberapa laptop berharga sangat tinggi, sedangkan MAE dan MAPE lebih "
                   "mencerminkan galat tipikal.")

    if pred_df is not None:
        st.divider()
        st.subheader("Aktual vs prediksi (data uji)")
        pm = [c.replace("Prediksi_", "") for c in pred_df.columns if c.startswith("Prediksi_")]
        sel = st.selectbox("Model", pm, index=pm.index(BEST_NAME) if BEST_NAME in pm else 0)
        pp = pd.DataFrame({"Aktual": pred_df["Harga_Aktual"] * RATE,
                           "Prediksi": pred_df[f"Prediksi_{sel}"] * RATE})
        pp["Galat"] = pp["Prediksi"] - pp["Aktual"]
        mx = float(max(pp["Aktual"].max(), pp["Prediksi"].max()))
        diag = alt.Chart(pd.DataFrame({"x": [0, mx], "y": [0, mx]})).mark_line(
            color="#d62728", strokeDash=[5, 5]).encode(x="x:Q", y="y:Q")
        sc = alt.Chart(pp).mark_circle(size=45, opacity=0.6).encode(
            x=alt.X("Aktual:Q", title=f"Harga aktual ({CUR})"),
            y=alt.Y("Prediksi:Q", title=f"Harga prediksi ({CUR})"),
            tooltip=[alt.Tooltip("Aktual:Q", format=",.0f"),
                     alt.Tooltip("Prediksi:Q", format=",.0f")])
        r1, r2 = st.columns(2)
        r1.altair_chart(sc + diag, **STRETCH)
        rh = alt.Chart(pp).mark_bar().encode(
            x=alt.X("Galat:Q", bin=alt.Bin(maxbins=30), title=f"Galat (prediksi − aktual, {CUR})"),
            y=alt.Y("count()", title="Jumlah"))
        r2.altair_chart(rh, **STRETCH)
        st.caption("Titik di dekat garis merah berarti prediksi mendekati harga aktual. "
                   "Sebaran melebar di harga tinggi menunjukkan galat besar di segmen premium.")


# ═════════════════════════════════════════════════════════════
# TAB 4 — FAKTOR HARGA
# ═════════════════════════════════════════════════════════════
with tab_feat:
    st.subheader("Fitur yang paling memengaruhi prediksi")
    if imp_df is None:
        st.info("`feature_importance.csv` tidak ditemukan.")
    else:
        sel = st.selectbox("Model", imp_df.columns.tolist(),
                           index=imp_df.columns.tolist().index(BEST_NAME)
                           if BEST_NAME in imp_df.columns else 0, key="imp_model")
        s = imp_df[sel].clip(lower=0)
        share = (s / s.sum() * 100).sort_values(ascending=False).rename("Kontribusi (%)")
        fi = share.reset_index().rename(columns={"index": "Fitur"})
        fi.columns = ["Fitur", "Kontribusi (%)"]
        ch = alt.Chart(fi).mark_bar().encode(
            x=alt.X("Kontribusi (%):Q"),
            y=alt.Y("Fitur:N", sort="-x", title=None),
            tooltip=["Fitur", alt.Tooltip("Kontribusi (%):Q", format=".1f")])
        st.altair_chart(ch, **STRETCH)
        st.caption("Nilai asli di `feature_importance.csv` dinormalisasi menjadi persentase "
                   "dari total per model, agar mudah dibandingkan. Makin besar, makin besar "
                   "pengaruh fitur tersebut pada prediksi model itu.")
        st.info("Ranking fitur berbeda antar model. Karena itu hasilnya dibaca sebagai "
                "indikasi pengaruh, bukan hubungan sebab-akibat. Jawaban atas pertanyaan "
                "\"apakah hardware memengaruhi harga\": ya, terutama spesifikasi CPU, GPU, "
                "dan RAM menonjol di semua model, meskipun urutan pastinya berbeda.")

    if coef_df is not None:
        with st.expander("Koefisien Multiple Linear Regression (10 terbesar dan terkecil)"):
            cf = coef_df.assign(koefisien=coef_df["koefisien"] * RATE).sort_values("koefisien", ascending=False)
            st.dataframe(pd.concat([cf.head(10), cf.tail(10)]), hide_index=True, **STRETCH)
            st.caption(f"Koefisien dalam {CUR}. Koefisien kategori langka (misalnya model GPU "
                       "yang hanya muncul beberapa kali) bisa sangat besar dan tidak stabil.")


# ═════════════════════════════════════════════════════════════
# TAB 5 — KEANDALAN MODEL
# ═════════════════════════════════════════════════════════════
with tab_rel:
    st.subheader("Seberapa andal estimasinya?")
    k1, k2 = st.columns(2)

    with k1:
        st.markdown(f"**Galat per segmen harga ({BEST_NAME})**")
        if seg_df is not None:
            seg_v = seg_df.assign(MAE=seg_df["MAE"] * RATE)
            sb = alt.Chart(seg_v).mark_bar().encode(
                x=alt.X("segmen:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("MAE:Q", title=f"MAE ({CUR})"),
                tooltip=["segmen", alt.Tooltip("MAE:Q", format=",.0f"), "MAPE_persen"])
            st.altair_chart(sb, **STRETCH)
            st.dataframe(seg_v.rename(columns={"MAE": f"MAE ({CUR})", "MAPE_persen": "MAPE (%)"}),
                         hide_index=True, **STRETCH)
            st.caption("Segmen = kuartil harga aktual pada data uji. Galat segmen termahal "
                       "sekitar enam kali lipat segmen lain.")

    with k2:
        st.markdown("**Overfitting: R² latih vs uji**")
        if comp_df is not None:
            ov = comp_df[["Model", "Train_R2", "Test_R2", "Gap_R2"]]
            ovl = ov.melt("Model", ["Train_R2", "Test_R2"], var_name="Data", value_name="R²")
            ovl["Data"] = ovl["Data"].map({"Train_R2": "Latih", "Test_R2": "Uji"})
            ob = alt.Chart(ovl).mark_bar().encode(
                x=alt.X("Model:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("R²:Q", scale=alt.Scale(domain=[0, 1])),
                xOffset="Data:N", color="Data:N")
            st.altair_chart(ob, **STRETCH)
            st.dataframe(ov, hide_index=True, **STRETCH)
            st.caption("Selisih R² (Gap) yang besar berarti model lebih menghafal data latih. "
                       "Random Forest paling besar, Multiple Linear Regression paling kecil.")

    if pred_df is not None:
        st.divider()
        st.markdown(f"**10 prediksi dengan galat terbesar ({BEST_NAME}, data uji)**")
        col = f"Prediksi_{BEST_NAME}"
        w = pred_df[["Harga_Aktual", col]].copy()
        w.columns = ["Aktual", "Prediksi"]
        w["Selisih"] = w["Prediksi"] - w["Aktual"]
        w["Selisih (%)"] = (w["Selisih"] / w["Aktual"] * 100).round(1)
        w = w.reindex(w["Selisih"].abs().sort_values(ascending=False).index).head(10)
        for c in ["Aktual", "Prediksi", "Selisih"]:
            w[c] = w[c].map(money)
        st.dataframe(w, hide_index=True, **STRETCH)

    st.divider()
    st.markdown("**Keterbatasan**")
    st.markdown(
        "- Data hanya **920 laptop**, dengan satu kali pembagian latih-uji (736 latih, 184 uji). "
        "Hasil bisa berubah bila pembagian berbeda.\n"
        "- Harga asli data dalam **INR**; tampilan rupiah hanyalah konversi kasar dengan kurs yang bisa diubah.\n"
        "- Estimasi **laptop premium kurang andal** karena datanya sedikit dan harganya ekstrem.\n"
        "- Ridge, Lasso, dan KNN **tidak diuji**; hanya tiga model yang dibandingkan.\n"
        "- Model hanya memakai spesifikasi hardware; merek, model spesifik, kondisi pasar, "
        "dan waktu pembelian tidak dimodelkan.")