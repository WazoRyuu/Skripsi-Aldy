"""Streamlit: cek peluang lanjut ke tahap interview (Tele Marketing) lewat input manual.

STANDALONE: tidak terhubung ke database/Google Sheet perusahaan, tidak menyimpan dan tidak mengirim data yang diisi.
Jalankan lokal:  streamlit run app.py
"""
import html
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
STEPS = ["Data diri", "Pendidikan", "Pengalaman", "Hasil"]

st.set_page_config(page_title="Cek Peluang Interview - PT XYZ Karir", layout="centered", initial_sidebar_state="collapsed")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root{--blue:#0a66e0;--blue-dark:#0852b4;--ink:#1b2430;--muted:#6b7685;--line:#d4dce5;--bg:#eaf2f7;--card:#ffffff;}
html, body, .stApp, .stApp *:not(svg):not(path):not([data-testid="stIconMaterial"]):not([data-testid="stExpanderIcon"]):not([class*="material"]){font-family:'Inter','Segoe UI',system-ui,-apple-system,Roboto,sans-serif;}
.stApp{background:var(--bg);color:var(--ink);}
header[data-testid="stHeader"], #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"]{display:none !important;}
.block-container{max-width:700px; padding:92px 16px 88px;}

/* bilah atas dan bawah */
.topbar{position:fixed; top:0; left:0; right:0; z-index:999; height:64px; background:#fff; border-bottom:1px solid #e3e9ef;
  display:flex; align-items:center; justify-content:space-between; padding:0 32px;}
.brand{display:flex; align-items:center; gap:14px;}
.brand .name{font-weight:700; font-size:20px; letter-spacing:.2px; color:var(--blue);}
.brand .sep{width:1px; height:22px; background:#c9d3de;}
.brand .sub{font-size:17px; color:#586578;}
.topbar .nav{font-size:14px; color:#3a4656;}
.sitefoot{position:fixed; left:0; right:0; bottom:0; z-index:999; height:44px; background:var(--blue); color:#fff;
  display:flex; align-items:center; justify-content:space-between; padding:0 32px; font-size:13px;}

/* judul halaman */
.crumb{font-size:13px; color:var(--muted); margin-bottom:6px;}
.crumb b{color:var(--blue); font-weight:500;}
.company{font-size:14px !important; color:#4a5667; margin:0;}
.title{font-size:26px; font-weight:600; color:var(--ink); margin:2px 0 18px; line-height:1.25;}

/* kartu formulir */
.st-key-card{background:var(--card); border-radius:18px; box-shadow:0 1px 2px rgba(20,40,70,.05); padding:26px 28px 20px;}
/* stepper */
.stepper{display:flex; align-items:center; margin:6px 0 26px;}
.stepper .s{display:flex; align-items:center; gap:8px; color:#9aa5b3; font-size:14px; white-space:nowrap;}
.stepper .s:not(:last-child)::after{content:""; flex:1; min-width:18px; height:1px; background:#d4dce5; margin:0 12px;}
.stepper .s{flex:0 1 auto;}
.stepper .s:not(:last-child){flex:1;}
.stepper .dot{width:24px; height:24px; border-radius:50%; background:#e3e8ee; color:#fff; display:inline-flex; align-items:center;
  justify-content:center; font-size:12px; font-weight:600; flex:none;}
.stepper .lbl{display:inline-block;}
.stepper .s.done .dot{background:var(--blue);}
.stepper .s.done{color:#38475a;}
.stepper .s.now .dot{background:var(--blue); box-shadow:0 0 0 4px #d9e8fc;}
.stepper .s.now{color:var(--ink); font-weight:600;}
.hint{font-size:13px !important; color:var(--muted); margin:0 0 6px;}
.req{color:#d92d20;}
.sect{font-size:15px; font-weight:600; color:var(--ink); margin:20px 0 6px; padding-top:14px; border-top:1px solid #edf1f5;}
.sect:first-of-type{border-top:none; padding-top:0; margin-top:4px;}

/* input */
[data-testid="stWidgetLabel"] p{font-size:14px !important; font-weight:500 !important; color:var(--ink) !important;}
.stRadio [role="radiogroup"]{gap:6px 22px;}
.stRadio label p{font-weight:400 !important;}

/* warna widget dipaksa terang agar terbaca walaupun browser/Streamlit memakai tema gelap */
.stApp{color-scheme:light;}
.stApp, .stApp p, .stApp label, .stApp span{--text-color:var(--ink);}
[data-testid="stSelectbox"] [role="group"], [data-testid="stNumberInputContainer"], [data-testid="stDateInputField"],
[data-testid="stTextInputRootElement"]{background:#fff !important; border:1px solid var(--line) !important; border-radius:10px !important; min-height:46px;}
[data-testid="stSelectbox"] [role="group"]:focus-within, [data-testid="stNumberInputContainer"]:focus-within, [data-testid="stDateInputField"]:focus-within,
[data-testid="stTextInputRootElement"]:focus-within{border-color:var(--blue) !important; box-shadow:0 0 0 3px #d9e8fc;}
[data-testid="stSelectbox"] input, [data-testid="stNumberInputField"], [data-testid="stDateInputField"] *, [data-testid="stTextInputRootElement"] input{
  color:var(--ink) !important; -webkit-text-fill-color:var(--ink) !important; caret-color:var(--blue);}
[data-testid="stSelectbox"] input::placeholder, [data-testid="stNumberInputField"]::placeholder, [data-testid="stTextInputRootElement"] input::placeholder{
  color:#8794a5 !important; -webkit-text-fill-color:#8794a5 !important; opacity:1;}
[data-testid="stSelectbox"] svg, [data-testid="stNumberInputContainer"] svg, [data-testid="stDateInputField"] svg{color:#5b6878 !important; fill:#5b6878 !important;}
[data-testid="stNumberInputContainer"] button{background:transparent !important; color:#5b6878 !important;}
[data-testid="stSelectbox"]:has([aria-disabled="true"]) [role="group"], [data-testid="stSelectbox"] [role="group"][data-disabled="true"]{background:#f3f6f9 !important;}
/* radio */
[data-testid="stRadioOption"] p{color:var(--ink) !important;}
[data-testid="stRadioOption"] > div > div:first-child{background:#fff !important; border:1.5px solid #9aa8b8 !important; border-radius:50%;}
[data-testid="stRadioOption"][data-selected="true"] > div > div:first-child{background:var(--blue) !important; border-color:var(--blue) !important;}
[data-testid="stRadioOption"][data-selected="true"] > div > div:first-child > div{background:#fff !important;}
/* daftar pilihan dropdown */
[data-testid="stSelectboxVirtualDropdown"]{background:#fff !important; border:1px solid var(--line); border-radius:12px; box-shadow:0 8px 24px rgba(20,40,70,.12);}
[data-testid="stSelectboxVirtualDropdown"] *{background-color:transparent; color:var(--ink) !important;}
[data-testid="stSelectboxVirtualDropdown"] [role="option"]{background:#fff !important;}
[data-testid="stSelectboxVirtualDropdown"] [role="option"][data-hovered="true"], [data-testid="stSelectboxVirtualDropdown"] [role="option"][data-focused="true"],
[data-testid="stSelectboxVirtualDropdown"] [role="option"][data-selected="true"]{background:#eaf2fc !important;}
/* kalender */
[role="dialog"]{background:#fff !important; color:var(--ink) !important;}
[role="dialog"] *{color:var(--ink);}

/* tombol */
.stButton > button{border-radius:999px; height:46px; font-weight:600; font-size:15px; padding:0 30px; box-shadow:none;}
.stButton > button[kind="primary"]{background:var(--blue); border:1px solid var(--blue); color:#fff;}
.stButton > button[kind="primary"]:hover{background:var(--blue-dark); border-color:var(--blue-dark);}
.stButton > button[kind="secondary"]{background:#fff; color:var(--blue); border:1px solid #b9cde6;}
.stButton > button[kind="secondary"]:hover{border-color:var(--blue); background:#f3f8fe;}

/* hasil */
.notice{background:#fff6e5; border:1px solid #f5d9a1; color:#7a4b00; border-radius:10px; padding:11px 14px; font-size:14px; margin:6px 0 12px;}
.result{border:1px solid var(--line); border-radius:14px; padding:20px 22px; margin:4px 0 14px;}
.badge{display:inline-block; font-size:13px; font-weight:600; border-radius:999px; padding:4px 12px; margin-bottom:10px;}
.result p{margin:0; font-size:15px; line-height:1.6; color:#2c3745;}
.tone-ok{background:#f2faf5; border-color:#bfe3cd;} .tone-ok .badge{background:#d8f0e1; color:#0b6b3a;}
.tone-info{background:#f2f7fe; border-color:#c3d8f6;} .tone-info .badge{background:#d9e8fc; color:#0a4fa8;}
.tone-warn{background:#fffaf0; border-color:#f3dcab;} .tone-warn .badge{background:#fbe8c1; color:#8a5a00;}
.tone-bad{background:#fdf5f4; border-color:#f1c9c4;} .tone-bad .badge{background:#f9dcd8; color:#a1281c;}
.tiles{display:grid; grid-template-columns:repeat(2,1fr); gap:10px; margin:0 0 14px;}
.tiles.four{grid-template-columns:repeat(4,1fr);}
.tile{border:1px solid #e3e9ef; border-radius:12px; padding:12px 14px; background:#fff;}
.tile span{display:block; font-size:12px; color:var(--muted); margin-bottom:4px;}
.tile b{font-size:18px; font-weight:600; color:var(--ink);}
.rcol{border:1px solid #e3e9ef; border-radius:12px; padding:14px 16px; margin-bottom:10px;}
.rcol h4{font-size:14px; font-weight:600; margin:0 0 8px; color:var(--ink);}
.rcol ul{margin:0; padding-left:18px;} .rcol li{font-size:14px; line-height:1.55; color:#2c3745; margin-bottom:5px;}
.disclaimer{font-size:12.5px; color:var(--muted); line-height:1.55; margin:8px 0 6px;}
table.sum{width:100%; border-collapse:collapse; font-size:14px;}
table.sum td{padding:8px 4px; border-bottom:1px solid #edf1f5; vertical-align:top;}
table.sum td:first-child{color:var(--muted); width:46%;}
[data-testid="stIconMaterial"],[data-testid="stExpanderIcon"],[class*="material-symbols"]{font-family:"Material Symbols Rounded","Material Icons" !important;}
[data-testid="stExpander"]{border:1px solid #e3e9ef !important; border-radius:12px !important;}

@media (max-width: 640px){
  .topbar, .sitefoot{padding:0 16px;} .topbar .nav{display:none;}
  .stepper .lbl{display:none;} .stepper .s.now .lbl{display:inline-block;}
  .tiles.four{grid-template-columns:repeat(2,1fr);} .title{font-size:22px;}
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
st.markdown('<div class="topbar"><div class="brand"><span class="name">PT XYZ</span><span class="sep"></span><span class="sub">Karir</span></div>'
            '<div class="nav">Cek Peluang Interview</div></div>'
            '<div class="sitefoot"><span>&copy; 2026 PT XYZ</span><span>Data yang diisi tidak disimpan</span></div>', unsafe_allow_html=True)


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


def verdict(p: float, thr: float):
    """(tone, judul, pesan) untuk pelamar, berdasarkan skor model."""
    if p >= thr + 0.15:
        return ("ok", "Peluang tinggi",
                "Profil Anda sangat sesuai dengan kriteria posisi Tele Marketing yang kami cari. "
                "Kemungkinan besar Anda akan dilanjutkan ke tahap interview.")
    if p >= thr:
        return ("info", "Peluang cukup baik",
                "Profil Anda cukup sesuai dengan kriteria posisi yang kami cari. Anda berpeluang dilanjutkan ke tahap interview.")
    if p >= thr - 0.10:
        return ("warn", "Peluang belum pasti",
                "Profil Anda berada di batas kriteria. Hasil akhirnya bergantung pada penilaian tim rekrutmen.")
    return ("bad", "Belum sesuai",
            "Mohon maaf, sayangnya skill dan pengalaman Anda belum sesuai dengan posisi yang sedang kami cari saat ini. "
            "Terima kasih sudah berminat, dan Anda dipersilakan mencoba kembali pada lowongan berikutnya.")


# ------------------------------------------------------------------------------------------------ state
S = st.session_state
DEFAULTS = {"step": 0, "f_tgl": None, "f_nikah": None, "f_prov": None, "f_kab": None, "f_kec": None,
            "f_pend": None, "f_skala": "Nilai (0-100)", "f_nilai": None, "f_ipk": None, "f_kuliah": "Tidak", "f_rencana": "Tidak",
            "f_terakhir": None, "f_bidang": None, "f_bagian": None, "f_level": None, "f_lama": None, "f_durasi": None, "f_jabatan": "",
            "f_sales": "Tidak", "f_lama_sales": None, "f_crew": "Tidak", "f_lama_crew": None, "f_mode": "Pelamar", "f_place": DEFAULT_PLACEMENT}
for _k, _v in DEFAULTS.items():
    S.setdefault(_k, _v)
for _k in [k for k in S if k.startswith("f_")]:  # menjaga nilai widget saat pindah langkah
    S[_k] = S[_k]


def reset_all():
    for k, v in DEFAULTS.items():
        S[k] = v


def go(step: int):
    S["step"] = step
    st.rerun()


def label(text, required=True):
    return f"{text} *" if required else text


def sect(title):
    st.markdown(f'<div class="sect">{html.escape(title)}</div>', unsafe_allow_html=True)


def stepper(cur: int):
    parts = []
    for i, name in enumerate(STEPS):
        cls = "now" if i == cur else ("done" if i < cur else "")
        parts.append(f'<div class="s {cls}"><span class="dot">{i + 1}</span><span class="lbl">{html.escape(name)}</span></div>')
    st.markdown(f'<div class="stepper">{"".join(parts)}</div>', unsafe_allow_html=True)


def nav(back: bool = True, nxt: str | None = "Lanjut"):
    """Tombol navigasi; mengembalikan (kembali, lanjut)."""
    c1, c2 = st.columns(2)
    b = c1.button("Kembali", key=f"back{S['step']}", width="stretch") if back else False
    n = c2.button(nxt, key=f"next{S['step']}", type="primary", width="stretch") if nxt else False
    return b, n


def need(items):
    miss = [name for name, val in items if val in (None, "")]
    if miss:
        st.markdown(f'<div class="notice">Lengkapi kolom yang wajib diisi: {html.escape(", ".join(miss))}.</div>', unsafe_allow_html=True)
    return not miss


# ------------------------------------------------------------------------------------------------ halaman
try:
    P = load_predictor()
except Exception as e:  # versi library tidak cocok dengan model, file model hilang, dll.
    st.error("Model tidak bisa dimuat. Pastikan `models/model_interview.joblib` ada dan versi library di `requirements.txt` "
             "sama dengan versi saat model dilatih.")
    st.exception(e)
    st.stop()

prov_list, reg_by_prov, kec_by_reg = load_regions()

st.markdown('<div class="crumb"><b>Beranda</b> &nbsp;&gt;&nbsp; Cek peluang interview</div>'
            '<p class="company">PT XYZ &middot; Tele Marketing</p>'
            '<div class="title">Cek peluang lanjut ke tahap interview</div>', unsafe_allow_html=True)

step = S["step"]
with st.container(key="card"):
    stepper(step)

    if step == 0:
        st.markdown('<p class="hint"><span class="req">*</span> menandakan kolom wajib diisi</p>', unsafe_allow_html=True)
        sect("Data diri")
        c1, c2 = st.columns(2)
        c1.date_input(label("Tanggal lahir"), key="f_tgl", min_value=date(1960, 1, 1), max_value=date(date.today().year - 16, 12, 31), format="DD/MM/YYYY")
        c2.selectbox(label("Status pernikahan"), NIKAH, index=None, placeholder="Pilih status", key="f_nikah")
        sect("Domisili saat ini")
        st.selectbox(label("Provinsi"), prov_list, index=None, placeholder="Pilih provinsi", key="f_prov")
        regs = reg_by_prov.get(S["f_prov"], [])
        if S["f_kab"] not in regs:
            S["f_kab"] = None
        st.selectbox(label("Kabupaten / Kota"), regs, index=None, placeholder="Pilih kabupaten atau kota", key="f_kab",
                     disabled=not S["f_prov"])
        kecs = kec_by_reg.get(S["f_kab"], [])
        if S["f_kec"] not in kecs:
            S["f_kec"] = None
        st.selectbox(label("Kecamatan"), kecs, index=None, placeholder="Pilih kecamatan", key="f_kec", disabled=not S["f_kab"])
        _, n = nav(back=False)
        if n:
            ok = need([("Tanggal lahir", S["f_tgl"]), ("Status pernikahan", S["f_nikah"]), ("Provinsi", S["f_prov"]),
                       ("Kabupaten / Kota", S["f_kab"]), ("Kecamatan", S["f_kec"])])
            if ok:
                go(1)

    elif step == 1:
        sect("Pendidikan")
        c1, c2 = st.columns(2)
        c1.selectbox(label("Pendidikan terakhir"), PEND, index=None, placeholder="Pilih pendidikan", key="f_pend")
        c2.radio("Skala nilai", ["Nilai (0-100)", "IPK (0-4)"], key="f_skala", horizontal=True)
        if S["f_skala"].startswith("Nilai"):
            st.number_input(label("Nilai terakhir", False), min_value=0.0, max_value=100.0, step=1.0, value=None, key="f_nilai",
                            placeholder="Opsional, kosongkan bila tidak tahu")
        else:
            st.number_input(label("IPK terakhir", False), min_value=0.0, max_value=4.0, step=0.01, value=None, key="f_ipk",
                            placeholder="Opsional, kosongkan bila tidak tahu")
        c1, c2 = st.columns(2)
        c1.radio("Apakah sedang berkuliah?", ["Tidak", "Ya"], key="f_kuliah", horizontal=True)
        c2.radio("Berencana berkuliah?", ["Tidak", "Ya"], key="f_rencana", horizontal=True, disabled=(S["f_kuliah"] == "Ya"))
        b, n = nav()
        if b:
            go(0)
        if n and need([("Pendidikan terakhir", S["f_pend"])]):
            go(2)

    elif step == 2:
        sect("Riwayat kerja")
        st.selectbox(label("Kapan terakhir bekerja?"), TERAKHIR, index=None, placeholder="Pilih jawaban", key="f_terakhir")
        c1, c2 = st.columns(2)
        c1.selectbox(label("Bidang perusahaan terakhir"), BIDANG, index=None, placeholder="Pilih bidang", key="f_bidang")
        c2.selectbox(label("Bagian / posisi terakhir"), BAGIAN, index=None, placeholder="Pilih posisi", key="f_bagian")
        c1, c2 = st.columns(2)
        c1.selectbox(label("Level jabatan terakhir"), LEVEL, index=None, placeholder="Pilih level", key="f_level")
        c2.selectbox(label("Lama bekerja di pekerjaan terakhir"), LAMA_KERJA, index=None, placeholder="Pilih durasi", key="f_lama")
        c1, c2 = st.columns(2)
        c1.selectbox(label("Durasi terlama di satu perusahaan"), DURASI, index=None, placeholder="Pilih durasi", key="f_durasi")
        c2.text_input(label("Jabatan yang paling lama dijalani", False), key="f_jabatan", placeholder="Contoh: Telesales, Kasir, Admin")
        sect("Pengalaman khusus")
        c1, c2 = st.columns(2)
        c1.radio("Pernah berpengalaman sales/marketing?", ["Tidak", "Ya"], key="f_sales", horizontal=True)
        c2.selectbox("Berapa lama (sales/marketing)?", LAMA_EXP, index=None, placeholder="Pilih durasi", key="f_lama_sales",
                     disabled=(S["f_sales"] == "Tidak"))
        c1, c2 = st.columns(2)
        c1.radio("Pernah menjadi crew store (Alfamart, Indomaret, dll)?", ["Tidak", "Ya"], key="f_crew", horizontal=True)
        c2.selectbox("Berapa lama (crew store)?", LAMA_EXP, index=None, placeholder="Pilih durasi", key="f_lama_crew",
                     disabled=(S["f_crew"] == "Tidak"))
        b, n = nav(nxt="Lihat hasil")
        if b:
            go(1)
        if n:
            items = [("Kapan terakhir bekerja", S["f_terakhir"]), ("Bidang perusahaan terakhir", S["f_bidang"]),
                     ("Bagian / posisi terakhir", S["f_bagian"]), ("Level jabatan terakhir", S["f_level"]),
                     ("Lama bekerja di pekerjaan terakhir", S["f_lama"]), ("Durasi terlama di satu perusahaan", S["f_durasi"])]
            if S["f_sales"] == "Ya":
                items.append(("Lama pengalaman sales/marketing", S["f_lama_sales"]))
            if S["f_crew"] == "Ya":
                items.append(("Lama pengalaman crew store", S["f_lama_crew"]))
            if need(items):
                go(3)

    else:  # hasil
        recruiter = S["f_mode"] == "Recruiter"
        placement = (S["f_place"] or DEFAULT_PLACEMENT).strip() or DEFAULT_PLACEMENT
        nilai = S["f_nilai"] if S["f_skala"].startswith("Nilai") else S["f_ipk"]
        row = {
            PREFIX["ts"]: pd.Timestamp.now(), PREFIX["tgl_lahir"]: pd.Timestamp(S["f_tgl"]), PREFIX["nikah"]: S["f_nikah"],
            PREFIX["prov"]: S["f_prov"], PREFIX["kab"]: S["f_kab"], PREFIX["kec"]: S["f_kec"],
            PREFIX["pend"]: S["f_pend"], PREFIX["nilai"]: "" if not nilai else str(nilai),
            PREFIX["bidang"]: S["f_bidang"], PREFIX["bagian"]: S["f_bagian"], PREFIX["level"]: S["f_level"],
            PREFIX["lama_kerja_terakhir"]: S["f_lama"], PREFIX["durasi_terlama"]: S["f_durasi"], PREFIX["jabatan_terlama"]: S["f_jabatan"],
            PREFIX["pernah_sales"]: S["f_sales"], PREFIX["lama_sales"]: S["f_lama_sales"] if S["f_sales"] == "Ya" else "< 1 tahun",
            PREFIX["pernah_crew"]: S["f_crew"], PREFIX["lama_crew"]: S["f_lama_crew"] if S["f_crew"] == "Ya" else "< 1 tahun",
            PREFIX["terakhir_kerja"]: S["f_terakhir"], PREFIX["kuliah"]: S["f_kuliah"],
            PREFIX["rencana_kuliah"]: "Sedang Berkuliah" if S["f_kuliah"] == "Ya" else S["f_rencana"],
            PREFIX["penempatan"]: placement,
        }
        res, err = None, None
        try:
            P.geo.placement_coord(placement)  # validasi dulu supaya pesan error jelas
            res = P.predict(pd.DataFrame([row]), placement=placement).iloc[0]
        except ValueError as e:  # lokasi penempatan tidak dikenal
            err = str(e)

        st.radio("Tampilan hasil", ["Pelamar", "Recruiter"], key="f_mode", horizontal=True,
                 help="Tampilan Recruiter menambahkan skor, tingkat keyakinan, dan alasan rinci.")
        if recruiter:
            st.text_input("Lokasi penempatan (nama kecamatan)", key="f_place")
        if err:
            st.markdown(f'<div class="notice">{html.escape(err)}</div>', unsafe_allow_html=True)
        else:
            p = float(res["AI_Skor"])
            tone, title, msg = verdict(p, P.thr)
            st.markdown(f'<div class="result tone-{tone}"><span class="badge">{html.escape(title)}</span><p>{html.escape(msg)}</p></div>',
                        unsafe_allow_html=True)
            d = res["AI_Jarak_km"]
            jarak = "-" if pd.isna(d) else f"{d:.0f} km"
            st.markdown(f'<div class="tiles"><div class="tile"><span>Jarak ke lokasi penempatan</span><b>{jarak}</b></div>'
                        f'<div class="tile"><span>Lokasi penempatan</span><b>{html.escape(placement)}</b></div></div>', unsafe_allow_html=True)

            if recruiter:
                st.markdown(
                    f'<div class="tiles four"><div class="tile"><span>Skor</span><b>{p:.2f}</b></div>'
                    f'<div class="tile"><span>Batas klasifikasi</span><b>{P.thr:.2f}</b></div>'
                    f'<div class="tile"><span>Rekomendasi</span><b style="font-size:14px">{html.escape(str(res["AI_Rekomendasi"]))}</b></div>'
                    f'<div class="tile"><span>Keyakinan</span><b style="font-size:14px">{html.escape(str(res["AI_Keyakinan"]))}</b></div></div>',
                    unsafe_allow_html=True)

                def ul(title_, text):
                    li = "".join(f"<li>{html.escape(x)}</li>" for x in str(text).split(" | ") if x.strip())
                    return f'<div class="rcol"><h4>{html.escape(title_)}</h4><ul>{li}</ul></div>' if li else ""

                st.markdown(ul("Alasan utama", res["AI_Alasan_Utama"]) + ul("Faktor lain yang menarik ke arah sebaliknya", res["AI_Faktor_Lain"])
                            + ul("Catatan kualitas data", res["AI_Catatan_Data"]), unsafe_allow_html=True)

        with st.expander("Ringkasan data yang Anda isi"):
            def fmt(v):
                return "-" if v in (None, "") else (v.strftime("%d/%m/%Y") if hasattr(v, "strftime") else str(v))
            rows = [("Tanggal lahir", S["f_tgl"]), ("Status pernikahan", S["f_nikah"]),
                    ("Domisili", ", ".join(x for x in [S["f_kec"], S["f_kab"], S["f_prov"]] if x)),
                    ("Pendidikan terakhir", S["f_pend"]), ("Nilai / IPK", nilai), ("Sedang berkuliah", S["f_kuliah"]),
                    ("Terakhir bekerja", S["f_terakhir"]), ("Bidang / posisi terakhir", f"{S['f_bidang']} / {S['f_bagian']}"),
                    ("Pengalaman sales/marketing", S["f_sales"]), ("Pengalaman crew store", S["f_crew"])]
            st.markdown('<table class="sum">' + "".join(f"<tr><td>{html.escape(a)}</td><td>{html.escape(fmt(b))}</td></tr>" for a, b in rows)
                        + "</table>", unsafe_allow_html=True)

        st.markdown('<p class="disclaimer">Hasil ini adalah estimasi otomatis berdasarkan pola data seleksi sebelumnya, bukan keputusan resmi '
                    'perekrutan. Penilaian akhir tetap dilakukan oleh tim rekrutmen.</p>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        if c1.button("Ubah data", key="edit", width="stretch"):
            go(0)
        if c2.button("Mulai ulang", key="restart", type="primary", width="stretch"):
            reset_all()
            st.rerun()
