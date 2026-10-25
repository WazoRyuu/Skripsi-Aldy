"""Cleaning + feature engineering. Dipakai SAMA PERSIS untuk training dan prediksi (anti train/serve skew)."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from geo import GeoResolver

# ---- pemetaan header Google Form -> nama pendek (cocok by prefix, tahan terhadap spasi/typo akhir) ----
HEADER_MAP = [
    ("timestamp", "ts"), ("nama lengkap", "nama"), ("nomor ktp", "ktp"), ("nomor hp", "hp"),
    ("alamat email", "email"), ("tanggal lahir", "tgl_lahir"), ("jenis kelamin", "jk"),
    ("status pernikahan", "nikah"), ("provinsi tinggal", "prov"), ("kabupaten", "kab"),
    ("kecamatan tinggal", "kec"), ("kelurahan tinggal", "kel"), ("alamat lengkap", "alamat"),
    ("pendidikan terakhir", "pend"), ("jurusan", "jurusan"), ("nama sekolah", "sekolah"),
    ("nilai terakhir", "nilai"), ("nama perusahaan", "perusahaan"), ("bidang perusahaan", "bidang"),
    ("pengalaman kerja 3 thn terakhir di bagian", "bagian"), ("pengalaman kerja 3 thn terakhir di level", "level"),
    ("berapa lama bekerja di pengalaman terakhir", "lama_kerja_terakhir"),
    ("jelaskan sekilas", "tanggung_jawab"), ("lampirkan curriculum", "cv"),
    ("dari semua pengalaman kerja anda berapa durasi", "durasi_terlama"),
    ("dari semua pengalaman kerja anda yang paling lama", "jabatan_terlama"),
    ("apakah pernah memiliki pengalaman sales", "pernah_sales"),
    ("berpengalaman sebagai sales", "lama_sales"),
    ("apakah pernah memiliki pengalaman crew", "pernah_crew"),
    ("berpengalaman sebagai crew", "lama_crew"),
    ("alasan mengapa", "alasan"), ("kapan anda terakhir bekerja", "terakhir_kerja"),
    ("apakah saat ini dalam kondisi hamil", "hamil"), ("apakah anda sedang berkuliah", "kuliah"),
    ("apakah anda memiliki rencana", "rencana_kuliah"), ("tempat lahir", "tempat_lahir"),
    ("lokasi penempatan", "penempatan"), ("lanjut interview", "label"),
    # ---- FORM BARU (form_baru/): header pendek & rapi; tetap dipetakan ke nama pendek yang sama ----
    ("nik ktp", "ktp"), ("provinsi domisili", "prov"), ("kecamatan domisili", "kec"), ("kelurahan/desa domisili", "kel"),
    ("nilai akhir", "nilai"), ("posisi/bagian terakhir", "bagian"), ("level jabatan terakhir", "level"),
    ("lama bekerja di pekerjaan terakhir", "lama_kerja_terakhir"), ("durasi terlama di satu perusahaan", "durasi_terlama"),
    ("jabatan terlama", "jabatan_terlama"), ("kapan terakhir bekerja", "terakhir_kerja"), ("sedang berkuliah", "kuliah"),
    ("berencana kuliah", "rencana_kuliah"), ("link cv", "cv"),
    ("pengalaman sales", "sales_exp"), ("pengalaman crew", "crew_exp"),
    ("pengalaman telemarketing", "exp_tele_raw"), ("pengalaman multifinance", "exp_fin_raw"),
    ("kemampuan microsoft office", "ms_office"), ("kesiapan bergabung", "siap_join"),
    ("status seleksi berkas", "status_berkas"), ("tanggal interview", "tgl_interview"), ("hasil interview", "hasil_interview"),
    ("catatan recruiter", "catatan_recruiter"),
]

# kolom yang benar-benar dipakai fitur model: hanya ini yang diperingatkan bila hilang (kolom lain boleh tidak ada)
MODEL_NEEDS = ["ts", "tgl_lahir", "nikah", "prov", "kab", "kec", "kel", "alamat", "pend", "nilai", "bidang", "bagian", "level",
               "lama_kerja_terakhir", "durasi_terlama", "jabatan_terlama", "pernah_sales", "lama_sales", "pernah_crew", "lama_crew",
               "terakhir_kerja", "kuliah", "rencana_kuliah"]

# ---- fitur yang masuk model ----
NUM_FEATURES = [
    "jarak_km", "usia", "nilai_norm", "lama_sales_ord", "lama_crew_ord", "lama_kerja_ord", "durasi_terlama_ord",
]
BIN_FEATURES = [
    "menikah", "punya_anak", "pernah_sales", "pernah_crew", "kuliah_sekarang", "rencana_kuliah",
    "jabatan_sales_like", "jabatan_tele_like", "jabatan_collection_like", "lokasi_presisi", "dekat_10km",
]
CAT_FEATURES = ["pend_level", "bidang", "bagian", "level", "terakhir_kerja"]
FEATURES = NUM_FEATURES + BIN_FEATURES + CAT_FEATURES

# Dikecualikan SENGAJA dari model: jenis kelamin, kehamilan, tempat lahir, nama/KTP/HP/email (identitas / diskriminatif).
# Usia & status pernikahan DIPAKAI karena merupakan persyaratan sebagian posisi PT XYZ (keputusan peneliti).


def _norm_header(h) -> str:
    return re.sub(r"\s+", " ", str(h)).strip().lower()


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    new, used = {}, set()
    for col in df.columns:
        h = _norm_header(col)
        for prefix, short in HEADER_MAP:
            if h.startswith(prefix) and short not in used:
                new[col] = short
                used.add(short)
                break
    return df.rename(columns=new)


def _s(x) -> str:
    return "" if pd.isna(x) else re.sub(r"\s+", " ", str(x)).strip()


def _yn(x) -> float:
    v = _s(x).lower()
    return 1.0 if v.startswith("ya") else 0.0 if v.startswith("tidak") else np.nan


def _ord_years(x) -> float:
    v = re.sub(r"^ya,?", "", _s(x).lower().replace(" ", ""))
    if not v:
        return np.nan
    if "fresh" in v or v.startswith("belumpernah"):
        return 0.0
    if v.startswith("<1"):
        return 1.0
    if v.startswith("1-3"):
        return 2.0
    if v.startswith(">2") or v.startswith(">3"):
        return 3.0
    return np.nan


def parse_nilai(x) -> float:
    """Nilai/IPK campur skala (4, 10, 100) -> skala 0-100."""
    v = _s(x).replace(",", ".")
    m = re.search(r"\d+(?:\.\d+)?", v)
    if not m:
        return np.nan
    f = float(m.group())
    if f <= 0:
        return np.nan
    if f <= 4.0:
        return f / 4 * 100
    if f <= 10:
        return f * 10
    return f if f <= 100 else np.nan


def pend_level(x) -> str:
    v = _s(x).lower()
    v = re.sub(r"\bstrata\s*([12])\b", r"s\1", v)
    v = re.sub(r"\bdiploma\s*(iv|iii|ii|i|[1-4])\b", lambda m: "d" + {"i": "1", "ii": "2", "iii": "3", "iv": "4"}.get(m.group(1), m.group(1)), v)
    v = re.sub(r"\b([sd])[\s\-.]+([1-4])\b", r"\1\2", v)  # "S-1", "S 1", "D 3" -> "s1", "d3"
    for key, lab in (("s3", "S2"), ("s2", "S2"), ("s1", "S1"), ("d4", "D4"), ("d3", "D3"), ("d1", "D1"), ("d2", "D2")):
        if re.search(rf"\b{key}\b", v):
            return lab
    if re.search(r"\b(sarjana|bachelor)\b", v):
        return "S1"
    if re.search(r"\b(magister|master)\b", v):
        return "S2"
    if "kuliah" in v or "mahasiswa" in v:
        return "Sedang Kuliah"
    if re.search(r"\b(sma|smk|smu|slta|stm|man|ma|smea|sma/smk|sma smk)\b", v) or "sma" in v or "smk" in v:
        return "SMA/SMK"
    if re.search(r"\b(smp|sltp|mts)\b", v) or "smp" in v:
        return "SMP"
    return "Lainnya"


def _kw(x, words) -> float:
    v = _s(x).lower()
    return float(any(w in v for w in words))


def _parse_dt(s) -> pd.Series:
    """ISO (2024-06-05 10:00:00) dibaca year-first; format lain (05/06/2024) dibaca day-first. Tanpa warning."""
    s = pd.Series(s) if not isinstance(s, pd.Series) else s
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    txt = s.astype(str).str.strip()
    iso = txt.str.match(r"^\d{4}-\d{1,2}-\d{1,2}")
    parts = []
    if iso.any():
        parts.append(pd.to_datetime(txt[iso], errors="coerce"))
    if (~iso).any():
        parts.append(pd.to_datetime(txt[~iso], errors="coerce", dayfirst=True))
    out = pd.concat(parts).reindex(s.index) if parts else pd.Series(pd.NaT, index=s.index)
    out = pd.to_datetime(out, errors="coerce")
    return out.where(out.dt.year.between(1900, 2100))


def build_features(raw: pd.DataFrame, geo: GeoResolver | None = None, placement_default: str = "Cisauk") -> pd.DataFrame:
    df = rename_columns(raw).copy()
    if "sales_exp" in df.columns:  # form baru: 1 pertanyaan "Tidak pernah / Ya, < 1 tahun / ..." -> pernah + lama
        df["pernah_sales"], df["lama_sales"] = df["sales_exp"], df["sales_exp"]
    if "crew_exp" in df.columns:
        df["pernah_crew"], df["lama_crew"] = df["crew_exp"], df["crew_exp"]
    missing = [short for short in MODEL_NEEDS if short not in df.columns]
    if missing:  # header form berubah / kolom hilang -> jangan crash, isi kosong (model tetap jalan dgn imputasi)
        import warnings
        warnings.warn(f"Kolom form tidak ditemukan, diisi kosong: {missing}")
        for m in missing:
            df[m] = np.nan
    for _p, short in HEADER_MAP:  # kolom lama yang tidak ada di form baru (jk, hamil, sekolah, ...) tidak dipakai model -> isi kosong
        if short not in df.columns and short != "label":
            df[short] = np.nan
    geo = geo or GeoResolver()
    out = pd.DataFrame(index=df.index)

    ts, lahir = _parse_dt(df.get("ts")), _parse_dt(df.get("tgl_lahir"))
    usia = ((ts.fillna(pd.Timestamp.today()) - lahir).dt.days / 365.25)
    out["usia"] = usia.where(usia.between(16, 65))

    nikah = df["nikah"].map(_s).str.lower()
    out["menikah"] = nikah.str.startswith(("menikah", "cerai")).astype(float)
    out["punya_anak"] = nikah.str.contains("memiliki anak") & ~nikah.str.contains("belum")
    out["punya_anak"] = out["punya_anak"].astype(float)

    # ---- domisili -> jarak ----
    geo_cols = ["prov", "kab", "kec", "kel", "alamat"]
    res = [geo.resolve(*r) for r in df.reindex(columns=geo_cols).itertuples(index=False, name=None)]
    g = pd.DataFrame(res, index=df.index)
    plc = df["penempatan"].map(_s).replace("", placement_default) if "penempatan" in df else placement_default
    plc = plc if isinstance(plc, pd.Series) else pd.Series(plc, index=df.index)
    out["jarak_km"] = [geo.distance_km(a, b, p) for a, b, p in zip(g["lat"], g["lng"], plc)]
    out["penempatan_ok"] = [0.0 if (isinstance(d, float) and np.isnan(d) and isinstance(a, float) and not np.isnan(a)) else 1.0
                            for d, a in zip(out["jarak_km"], g["lat"])]
    out["lokasi_presisi"] = (g["level"] == "kecamatan").astype(float)
    out["dekat_10km"] = (out["jarak_km"] <= 10).astype(float)
    out.loc[out["jarak_km"].isna(), "dekat_10km"] = np.nan
    out["geo_match"], out["geo_level"], out["geo_conf"] = g["match"], g["level"], g["conf"]

    out["pend_level"] = df["pend"].map(pend_level)
    out["nilai_norm"] = df["nilai"].map(parse_nilai)
    for c in ("bidang", "bagian", "level", "terakhir_kerja"):
        out[c] = df[c].map(_s).replace("", "Tidak diisi")

    out["lama_sales_ord"] = df["lama_sales"].map(_ord_years)
    out["lama_crew_ord"] = df["lama_crew"].map(_ord_years)
    out["lama_kerja_ord"] = df["lama_kerja_terakhir"].map(_ord_years)
    out["durasi_terlama_ord"] = df["durasi_terlama"].map(_ord_years)
    out["pernah_sales"] = df["pernah_sales"].map(_yn)
    out["pernah_crew"] = df["pernah_crew"].map(_yn)
    # lama sales/crew tidak bermakna kalau jawaban "Tidak" (form tetap mengisi default "< 1 tahun")
    out.loc[out["pernah_sales"] == 0, "lama_sales_ord"] = 0.0
    out.loc[out["pernah_crew"] == 0, "lama_crew_ord"] = 0.0
    out["kuliah_sekarang"] = df["kuliah"].map(_yn)
    out["rencana_kuliah"] = df["rencana_kuliah"].map(lambda x: 1.0 if _s(x).lower().startswith(("ya", "sedang")) else 0.0)

    jab = df["jabatan_terlama"].map(_s) + " " + df["bagian"].map(_s)
    out["jabatan_sales_like"] = jab.map(lambda x: _kw(x, ["sales", "marketing", "sale ", "spg", "spb", "canvass", "promotor", "account officer"]))
    out["jabatan_tele_like"] = jab.map(lambda x: _kw(x, ["tele", "call center", "customer service", "cs ", "csr", "contact center"]))
    out["jabatan_collection_like"] = jab.map(lambda x: _kw(x, ["collect", "penagih", "remedial"]))

    # meta (bukan fitur)
    for c in ("ts", "nama", "ktp", "hp", "email", "label"):
        if c in df:
            out["_" + c] = df[c]
    return out


def to_label(s: pd.Series) -> pd.Series:
    return s.map(lambda x: 1 if str(x).strip().lower() in ("true", "1", "ya", "yes", "lolos") else 0)


def dedupe_applicants(feat: pd.DataFrame) -> pd.DataFrame:
    """Satu baris per pelamar (KTP). Ambil submit terakhir; label = 1 jika pernah lolos pada submit mana pun."""
    key = feat["_ktp"].map(lambda x: re.sub(r"\D", "", str(x)))
    key = key.where(key.str.len() >= 10, feat["_nama"].map(_s).str.lower() + "|" + feat["_hp"].map(lambda x: re.sub(r"\D", "", str(x))))
    f = feat.assign(_key=key, _y=to_label(feat["_label"]))
    f = f.sort_values("_ts")
    ymax = f.groupby("_key")["_y"].max()
    last = f.groupby("_key", as_index=False).tail(1).copy()
    last["_y"] = last["_key"].map(ymax)
    return last.sort_values("_ts").reset_index(drop=True)
