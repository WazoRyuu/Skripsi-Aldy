"""Streamlit: cek peluang lolos interview (Tele Marketing) lewat input manual.

STANDALONE: tidak terhubung ke database/Google Sheet perusahaan, tidak menyimpan dan tidak mengirim data yang diisi.
Jalankan lokal:  streamlit run app.py
"""
import sys
import warnings
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from features import HEADER_MAP  # noqa: E402
from predict import Predictor  # noqa: E402

DEFAULT_PLACEMENT = "Cisauk"
PREFIX = {short: prefix for prefix, short in HEADER_MAP}  # nama pendek -> awalan header form (dikenali build_features)

PEND = ["SMA/SMK", "D1", "D3", "D4", "S1", "S2", "Sedang Berkuliah"]
NIKAH = ["Belum Menikah", "Menikah Belum Memiliki Anak", "Menikah Sudah Memiliki Anak", "Cerai Memiliki Anak"]
BIDANG = ["Fresh Graduated", "Lembaga Keuangan (Perbankan & Non-Perbankan)", "Retail/Perdagangan", "Telekomunikasi", "Jasa",
          "E-Commerce", "Food and Beverage", "Outsourcing", "Manufaktur", "Pemerintahan/BUMN", "Ekspedisi/Transportasi",
          "Distributor", "Pendidikan", "Konstruksi", "Kesehatan"]
BAGIAN = ["Fresh Graduated/Belum Pernah Bekerja", "Tele Marketing/Tele Collection/Tele Operation", "Sales/Marketing", "Admin",
          "Crew Store", "Front Liner", "Collertor/Tenaga Penagih", "Accounting", "Tenaga Gudang/Helper", "Driver", "IT"]
LEVEL = ["Fresh Graduated", "Staff", "SPV", "Dept Head", "General Manager"]
LAMA_KERJA = ["< 1 tahun", "1 - 3 tahun", "> 2 tahun"]
DURASI = ["Fresh Graduated", "< 1 tahun", "1 - 3 tahun", "> 3 tahun"]
TERAKHIR = ["Saya saat ini masih aktif bekerja", "3 - 6 bulan yang lalu", "> 3 bulan yang lalu", "> 6 bulan", "Fresh Graduated"]
LAMA_EXP = ["< 1 tahun", "1 - 3 tahun", "> 3 tahun"]

st.set_page_config(page_title="Cek Peluang Interview", page_icon="📋", layout="centered")


@st.cache_resource(show_spinner="Memuat model...")
def load_predictor():
    return Predictor()


@st.cache_data
def load_regions():
    kec = pd.read_csv(ROOT / "data" / "kec_coords.csv", dtype={"kode": str})
    reg = pd.read_csv(ROOT / "data" / "regions_l12.csv", dtype={"kode": str})
    names = dict(zip(reg["kode"], reg["nama"]))
    kec["reg"] = kec["kode"].str[:5]
    kec["prov"] = kec["kode"].str[:2]
    prov = sorted(names[p] for p in kec["prov"].unique() if p in names)
    prov_code = {names[p]: p for p in kec["prov"].unique() if p in names}
    reg_by_prov = {pn: sorted(names[r] for r in kec.loc[kec["prov"] == pc, "reg"].unique() if r in names)
                   for pn, pc in prov_code.items()}
    reg_code = {names[r]: r for r in kec["reg"].unique() if r in names}
    kec_by_reg = {rn: sorted(kec.loc[kec["reg"] == rc, "nama"].tolist()) for rn, rc in reg_code.items()}
    return prov, reg_by_prov, kec_by_reg


def pick(options, default):
    return options.index(default) if default in options else 0


def verdict(p: float, thr: float):
    """(level, judul, pesan) untuk pelamar, berdasarkan skor model."""
    if p >= thr + 0.15:
        return ("success", "Peluang tinggi",
                "Profil Anda sangat sesuai dengan kriteria posisi Tele Marketing yang kami cari. "
                "Kemungkinan besar Anda akan dilanjutkan ke tahap interview.")
    if p >= thr:
        return ("info", "Peluang cukup baik",
                "Profil Anda cukup sesuai dengan kriteria posisi yang kami cari. Anda berpeluang dilanjutkan ke tahap interview.")
    if p >= thr - 0.10:
        return ("warning", "Peluang belum pasti",
                "Profil Anda berada di batas kriteria. Hasil akhirnya bergantung pada penilaian tim rekrutmen.")
    return ("error", "Belum sesuai",
            "Mohon maaf, sayangnya skill dan pengalaman Anda belum sesuai dengan posisi yang sedang kami cari saat ini. "
            "Terima kasih sudah berminat, dan Anda dipersilakan mencoba kembali pada lowongan berikutnya.")


# ------------------------------------------------------------------------------------------------ UI
st.title("📋 Cek Peluang Lolos Interview")
st.caption("Posisi: Tele Marketing · Isi data di bawah untuk melihat perkiraan peluang Anda dilanjutkan ke tahap interview.")

with st.sidebar:
    st.header("Pengaturan")
    mode = st.radio("Tampilan", ["Pelamar", "Recruiter"], help="Recruiter menampilkan skor dan alasan rinci.")
    placement = st.text_input("Lokasi penempatan (nama kecamatan)", DEFAULT_PLACEMENT)
    st.info("Aplikasi ini **berdiri sendiri**: data yang Anda isi tidak disimpan dan tidak dikirim ke database perusahaan.")

try:
    P = load_predictor()
except Exception as e:  # versi library tidak cocok dengan model, file model hilang, dll.
    st.error("Model tidak bisa dimuat. Pastikan `models/model_interview.joblib` ada dan versi library di `requirements.txt` "
             "sama dengan versi saat model dilatih.")
    st.exception(e)
    st.stop()

prov_list, reg_by_prov, kec_by_reg = load_regions()

st.subheader("1. Data diri")
c1, c2 = st.columns(2)
tgl_lahir = c1.date_input("Tanggal lahir", value=date(2000, 1, 1), min_value=date(1960, 1, 1), max_value=date.today())
nikah = c2.selectbox("Status pernikahan", NIKAH)

st.subheader("2. Domisili (tempat tinggal saat ini)")
prov = st.selectbox("Provinsi", prov_list, index=pick(prov_list, "Banten"))
regs = reg_by_prov[prov]
reg = st.selectbox("Kabupaten / Kota", regs, index=pick(regs, "Kabupaten Tangerang"))
kecs = kec_by_reg[reg]
kec = st.selectbox("Kecamatan", kecs, index=pick(kecs, "Cisauk"))

st.subheader("3. Pendidikan")
c1, c2 = st.columns(2)
pend = c1.selectbox("Pendidikan terakhir", PEND, index=pick(PEND, "SMA/SMK"))
skala = c2.radio("Skala nilai", ["Nilai (0–100)", "IPK (0–4)"], horizontal=True)
nilai = st.number_input("Nilai terakhir / IPK (isi 0 bila tidak tahu)", min_value=0.0,
                        max_value=100.0 if skala.startswith("Nilai") else 4.0,
                        value=80.0 if skala.startswith("Nilai") else 0.0, step=0.1 if skala.startswith("IPK") else 1.0)
c1, c2 = st.columns(2)
kuliah = c1.radio("Apakah sedang berkuliah?", ["Tidak", "Ya"], horizontal=True)
rencana = c2.radio("Berencana berkuliah?", ["Tidak", "Ya"], horizontal=True, disabled=(kuliah == "Ya"))

st.subheader("4. Riwayat kerja")
terakhir = st.selectbox("Kapan terakhir bekerja?", TERAKHIR)
c1, c2 = st.columns(2)
bidang = c1.selectbox("Bidang perusahaan tempat kerja terakhir", BIDANG)
bagian = c2.selectbox("Bagian / posisi di pekerjaan terakhir", BAGIAN)
c1, c2 = st.columns(2)
level = c1.selectbox("Level jabatan terakhir", LEVEL)
lama_kerja = c2.selectbox("Lama bekerja di pekerjaan terakhir", LAMA_KERJA)
c1, c2 = st.columns(2)
durasi = c1.selectbox("Durasi terlama bekerja di satu perusahaan", DURASI)
jabatan = c2.text_input("Jabatan yang paling lama dijalani", placeholder="mis. Telesales, Kasir, Admin")

st.subheader("5. Pengalaman khusus")
c1, c2 = st.columns(2)
pernah_sales = c1.radio("Pernah berpengalaman sales/marketing?", ["Tidak", "Ya"], horizontal=True)
lama_sales = c2.selectbox("Berapa lama (sales/marketing)?", LAMA_EXP, disabled=(pernah_sales == "Tidak"))
c1, c2 = st.columns(2)
pernah_crew = c1.radio("Pernah menjadi crew store (Alfamart/Indomaret dll)?", ["Tidak", "Ya"], horizontal=True)
lama_crew = c2.selectbox("Berapa lama (crew store)?", LAMA_EXP, disabled=(pernah_crew == "Tidak"))

go = st.button("Cek peluang saya", type="primary", use_container_width=True)

if go:
    nilai_txt = "" if nilai == 0 else str(nilai)
    row = {
        PREFIX["ts"]: pd.Timestamp.now(), PREFIX["tgl_lahir"]: pd.Timestamp(tgl_lahir), PREFIX["nikah"]: nikah,
        PREFIX["prov"]: prov, PREFIX["kab"]: reg, PREFIX["kec"]: kec,
        PREFIX["pend"]: pend, PREFIX["nilai"]: nilai_txt,
        PREFIX["bidang"]: bidang, PREFIX["bagian"]: bagian, PREFIX["level"]: level,
        PREFIX["lama_kerja_terakhir"]: lama_kerja, PREFIX["durasi_terlama"]: durasi, PREFIX["jabatan_terlama"]: jabatan,
        PREFIX["pernah_sales"]: pernah_sales, PREFIX["lama_sales"]: lama_sales if pernah_sales == "Ya" else "< 1 tahun",
        PREFIX["pernah_crew"]: pernah_crew, PREFIX["lama_crew"]: lama_crew if pernah_crew == "Ya" else "< 1 tahun",
        PREFIX["terakhir_kerja"]: terakhir, PREFIX["kuliah"]: kuliah,
        PREFIX["rencana_kuliah"]: "Sedang Berkuliah" if kuliah == "Ya" else rencana,
        PREFIX["penempatan"]: placement or DEFAULT_PLACEMENT,
    }
    try:
        res = P.predict(pd.DataFrame([row]), placement=placement or DEFAULT_PLACEMENT).iloc[0]
    except ValueError as e:  # lokasi penempatan tidak dikenal
        st.error(str(e))
        st.stop()

    p = float(res["AI_Skor"])
    level_, title, msg = verdict(p, P.thr)
    getattr(st, level_)(f"**{title}**\n\n{msg}")

    d = res["AI_Jarak_km"]
    c1, c2 = st.columns(2)
    c1.metric("Jarak ke lokasi penempatan", "-" if pd.isna(d) else f"±{d:.0f} km", help="Garis lurus dari pusat kecamatan.")
    c2.metric("Lokasi penempatan", placement or DEFAULT_PLACEMENT)

    if mode == "Recruiter":
        st.divider()
        st.markdown(f"**Skor:** {p:.2f} (batas {P.thr:.2f}) · **Rekomendasi:** {res['AI_Rekomendasi']} · **Keyakinan:** {res['AI_Keyakinan']}")
        st.markdown("**Alasan utama**")
        for r in str(res["AI_Alasan_Utama"]).split(" | "):
            st.markdown(f"- {r}")
        if res["AI_Faktor_Lain"]:
            st.markdown("**Faktor lain yang menarik ke arah sebaliknya**")
            for r in str(res["AI_Faktor_Lain"]).split(" | "):
                st.markdown(f"- {r}")

    st.caption("Hasil ini adalah estimasi otomatis berdasarkan pola data seleksi sebelumnya, bukan keputusan resmi perekrutan. "
               "Penilaian akhir tetap dilakukan oleh tim rekrutmen.")
