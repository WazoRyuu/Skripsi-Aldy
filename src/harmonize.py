"""Menyamakan 3 generasi form pelamar (header berbeda-beda) ke SATU skema kanonik, khusus Tele Marketing.

  v1  Form_Loker_Tele_Marketing_Jawaban   (2024-06 .. 2026-01)  header paling lengkap, domisili terstruktur
  v2  "Kuis tanpa judul" (MS Forms)        (2026-03 .. )         pertanyaan pengalaman tele/multifinance, alamat bebas + jarak (pilihan)
  v3  Form Data Diri Kandidat (MS Forms)   (2026-07 .. )         banyak posisi (disaring Tele Marketing), riwayat kerja teks bebas

Hanya informasi yang MASIH BERARTI SAMA di tiga form yang dijadikan fitur inti (CORE). Sisanya disimpan sebagai kolom ext_*
(hanya terisi di era tertentu) supaya tidak hilang, tetapi tidak dipakai model gabungan.

Label: pelamar ada di Rekap_All (unit Tele Marketing) = lolos seleksi berkas & lanjut interview. Dicocokkan lewat
No HP + kemiripan nama, atau nama persis. Tanggal interview di rekap dipakai untuk membedakan "melamar lagi setelah pernah
interview" (label tidak jelas -> dikeluarkan) dari lamaran yang memang berujung interview.
"""
from __future__ import annotations

import re
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from features import build_features, pend_level
from geo import GeoResolver
from labeling import norm_hp, norm_name

CORE = ["usia", "pend_level", "fresh_grad", "exp_tele", "exp_tele_ord", "exp_finance", "exp_finance_ord", "jarak_km", "lokasi_presisi"]
DEFAULT_PLACEMENT = "Cisauk"

# ------------------------------------------------------------------ util parsing
_MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "mei": 5, "may": 5, "jun": 6, "jul": 7, "agu": 8, "aug": 8, "ags": 8,
           "sep": 9, "okt": 10, "oct": 10, "nov": 11, "des": 12, "dec": 12}


def _s(x) -> str:
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) or (x is pd.NaT) else re.sub(r"\s+", " ", str(x)).strip()


def parse_dob(x, ref=None) -> pd.Timestamp:
    """Tanggal lahir: datetime, 01-05-1998, 09/11/1992, 1998-05-01, salah ketik '21-09-20000' -> Timestamp / NaT."""
    if isinstance(x, (pd.Timestamp, datetime)):
        return pd.Timestamp(x) if 1930 <= x.year <= 2015 else pd.NaT
    v = _s(x)
    if not v:
        return pd.NaT
    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})", v)
    if m:
        y, mo, d = map(int, m.groups())
    else:
        m = re.match(r"^(\d{1,2})\D+(\d{1,2})\D+(\d{4})\d*$", v)
        if not m:
            return pd.NaT
        d, mo, y = map(int, m.groups())
    try:
        return pd.Timestamp(year=y, month=mo, day=d) if 1930 <= y <= 2015 else pd.NaT
    except ValueError:
        return pd.NaT


def usia_from(dob: pd.Timestamp, ts: pd.Timestamp) -> float:
    if pd.isna(dob) or pd.isna(ts):
        return np.nan
    u = (ts - dob).days / 365.25
    return u if 16 <= u <= 65 else np.nan


_BR_USIA = [("<19", 18.0), ("19", 22.0), ("26", 30.5), (">35", 40.0)]


def usia_bracket(x) -> float:
    v = _s(x).replace(" ", "")
    if v.startswith("<"):
        return 18.0
    if v.startswith(">"):
        return 40.0
    if v.startswith("19"):
        return 22.0
    if v.startswith("26"):
        return 30.5
    return np.nan


def jarak_bracket(x) -> float:
    """'1 - 15 KM' -> titik tengah (km)."""
    v = _s(x).lower().replace(" ", "")
    if v.startswith(">"):
        return 55.0
    m = re.match(r"(\d+)-(\d+)", v)
    return (int(m.group(1)) + int(m.group(2))) / 2 if m else np.nan


def months_between(text, ref: pd.Timestamp) -> float:
    """Durasi (bulan) dari teks periode kerja bebas: '07 Agustus 2023 - 31 Desember 2025', '6 Januari 2025 - Sekarang',
    '1 tahun 8 bulan', '01/10/2025- 01/09/2026'. NaN bila tidak bisa dibaca."""
    v = _s(text).lower()
    if not v or v in {"-", "tidak ada"}:
        return np.nan
    m = re.search(r"(\d+)\s*(?:tahun|thn|th)\b(?:\s*(\d+)\s*bulan)?", v)
    if m:
        return int(m.group(1)) * 12 + int(m.group(2) or 0)
    m = re.fullmatch(r"(\d+)\s*(?:bulan|bln)", v)
    if m:
        return int(m.group(1))
    pat = re.compile(r"(?:(\d{1,2})\s*)?(jan\w*|feb\w*|mar\w*|apr\w*|mei|may|jun\w*|jul\w*|ag\w*|aug\w*|sep\w*|okt\w*|oct\w*|nov\w*|des\w*|dec\w*)"
                     r"\s*(\d{4})?|(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})|(sekarang|saat ini|present|now|masih)")
    pts = []
    for g in pat.finditer(v):
        if g.group(7):
            pts.append(("now", None, None))
        elif g.group(4):
            pts.append((int(g.group(6)), int(g.group(5)), None))
        elif g.group(2) and g.group(2)[:3] in _MONTHS:
            pts.append((int(g.group(3)) if g.group(3) else None, _MONTHS[g.group(2)[:3]], None))
    if len(pts) < 2:
        return np.nan
    (y1, m1, _), (y2, m2, _) = pts[0], pts[-1]
    if y2 == "now":
        y2, m2 = ref.year, ref.month
    if y1 == "now":
        return np.nan
    if y1 is None and y2 is None:
        return np.nan
    if y2 is None:
        y2 = y1 + (1 if m2 < m1 else 0)
    if y1 is None:
        y1 = y2 - (1 if m1 > m2 else 0)
    mo = (y2 - y1) * 12 + (m2 - m1)
    return float(mo) if 0 <= mo <= 600 else np.nan


def _ord(months: float, has: bool) -> float:
    """0 tidak ada, 1 < 1 tahun, 2 >= 1 tahun, NaN = ada tapi lama tak diketahui."""
    if not has:
        return 0.0
    if np.isnan(months):
        return np.nan
    return 1.0 if months < 12 else 2.0


TELE_KW = ["tele", "call center", "callcenter", "contact center", "customer service", "customer care", "csr", "collection",
           "collector", "kol ", "kolektor", "verifikator", "verifikasi", "desk", "penagih", "telesales"]
FIN_KW = ["bank", "finance", "multifinance", "leasing", "bfi", "adira", "fif", "home credit", "kredit", "pembiayaan", "pnm", "mega ",
          "bca", "bri", "bni", "mandiri", "cimb", "ocbc", "permata", "btn", "danamon", "koperasi", "asuransi", "sinarmas", "oto ",
          "acc ", "wom ", "maybank", "uob", "kredivo", "akulaku", "indodana", "pinjaman", "pinjol"]
SALES_KW = ["sales", "marketing", "spg", "spb", "canvass", "promotor", "account officer", "ao ", "business development"]
NONE_TXT = {"", "-", "--", "tidak ada", "tidak", "belum ada", "belum bekerja", "nan", "none", "n/a", "na", "gak ada", "ga ada"}


def _has(text: str, kws) -> bool:
    t = " " + _s(text).lower() + " "
    return any(k in t for k in kws)


def _pend(x) -> str:
    v = _s(x).lower()
    if re.search(r"paket\s*c|\bsederajat\b", v):
        return "SMA/SMK"
    return pend_level(x)


_STOPW = r"(?:kab|kabupaten|kota|kel|kelurahan|desa|ds|kec|kecamatan|rt|rw|prov|provinsi|banten|jawa|dki|jakarta|kode|no|blok)"


def split_address(text: str) -> dict:
    """Ambil 'kec ...', 'kab/kota ...', 'kel/desa ...' dari alamat bebas -> petunjuk terstruktur untuk GeoResolver."""
    t = " " + re.sub(r"[,;/]", " , ", _s(text).lower()) + " "
    t = re.sub(r"\b(kec|kab|kel|ds)\.", r"\1 ", t)

    def grab(keys):
        m = re.search(rf"\b(?:{keys})\s+([a-z][a-z ]*?)(?=\s+{_STOPW}\b|\s*,|\s*\d|\s*$)", t)
        return m.group(1).strip() if m else ""

    return {"kec": grab("kecamatan|kec"), "kab": grab("kabupaten|kab|kota"), "kel": grab("kelurahan|kel|desa|ds")}


# ------------------------------------------------------------------ geo helper
class Geo:
    def __init__(self):
        self.g = GeoResolver()

    def km(self, prov="", kab="", kec="", kel="", alamat="", alamat_only=False):
        if alamat_only:  # hanya alamat bebas: petakan "kec X kab Y" di dalam teks ke kolom terstruktur
            h = split_address(alamat)
            kab, kec, kel = kab or h["kab"], kec or h["kec"], kel or h["kel"]
        r = self.g.resolve(prov, kab, kec, kel, alamat, alamat_w=0.9 if alamat_only else 0.4)
        d = self.g.distance_km(r["lat"], r["lng"], DEFAULT_PLACEMENT)
        return d, r["level"], r["conf"], r["match"]


def _geo_cols(km, level, conf, match, bracket_mid=np.nan):
    """Gabungkan hasil geocoding dengan jarak yang dilaporkan sendiri (bracket) bila ada.
    Bracket dipakai sebagai pengaman: geocoding yang jauh melenceng dari bracket dianggap salah baca."""
    src = {"kecamatan": "kecamatan", "kabupaten_kota": "kab/kota", "tidak_terbaca": "tidak_terbaca"}[level]
    if not np.isnan(bracket_mid):
        lo, hi = {8.0: (0, 15), 23.0: (15, 30), 38.0: (30, 45), 55.0: (45, 9999)}.get(bracket_mid, (0, 9999))
        if level == "tidak_terbaca" or np.isnan(km) or not (lo * 0.6 - 3 <= km <= hi * 1.5 + 3):
            return bracket_mid, "bracket", 0.0, "", "bracket"
        return km, src, conf, match, src
    return (km, src, conf, match, src) if level != "tidak_terbaca" else (np.nan, src, 0.0, "", src)


# ------------------------------------------------------------------ adapters: tiap form -> kolom kanonik
def adapt_v1(raw: pd.DataFrame, geo: Geo) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = build_features(raw, geo.g, DEFAULT_PLACEMENT)
    out = pd.DataFrame(index=f.index)
    out["era"] = "v1"
    out["ts"], out["nama"], out["hp"], out["ktp"] = pd.to_datetime(f["_ts"], errors="coerce"), f["_nama"], f["_hp"], f["_ktp"]
    out["usia"], out["pend_level"] = f["usia"], f["pend_level"].replace({"Sedang Kuliah": "Lainnya"})
    bagian = f["bagian"].str.lower()
    fresh = (f["bidang"].str.contains("fresh", case=False) | f["level"].str.contains("fresh", case=False) | bagian.str.contains("fresh"))
    out["fresh_grad"] = fresh.astype(float)
    tele = bagian.str.contains("tele") | (f["jabatan_tele_like"] == 1) | (f["jabatan_collection_like"] == 1)
    fin = f["bidang"].str.contains("keuangan", case=False)
    ten = f["lama_kerja_ord"].map(lambda v: np.nan if pd.isna(v) else (1.0 if v <= 1 else 2.0))
    out["exp_tele"] = (tele & ~fresh).astype(float)
    out["exp_tele_ord"] = [(_t if h else 0.0) for _t, h in zip(ten, out["exp_tele"] == 1)]
    out["exp_finance"] = (fin & ~fresh).astype(float)
    out["exp_finance_ord"] = [(_t if h else 0.0) for _t, h in zip(ten, out["exp_finance"] == 1)]
    out["exp_sales"] = ((f["pernah_sales"] == 1) | (f["jabatan_sales_like"] == 1)).astype(float)
    lv = f["geo_level"].map({"kecamatan": "kecamatan", "kabupaten_kota": "kab/kota", "tidak_terbaca": "tidak_terbaca"})
    out["jarak_km"], out["geo_level"], out["geo_conf"], out["geo_match"], out["jarak_src"] = f["jarak_km"], lv, f["geo_conf"], f["geo_match"], lv
    for c in ("menikah", "punya_anak", "nilai_norm", "kuliah_sekarang"):
        out["ext_" + c] = f[c]
    return out


def adapt_v2(raw: pd.DataFrame, geo: Geo) -> pd.DataFrame:
    c = {re.sub(r"\s+", " ", str(k)).strip().lower(): k for k in raw.columns}

    def col(prefix):
        k = next(v for kk, v in c.items() if kk.startswith(prefix))
        return raw[k]

    ts = pd.to_datetime(col("waktu mulai"), errors="coerce")
    dob = col("tanggal lahir").map(parse_dob)
    usia = [usia_from(d, t) for d, t in zip(dob, ts)]
    usia = [u if not np.isnan(u) else usia_bracket(b) for u, b in zip(usia, col("usia anda"))]
    pend = col("pendidikan terakhir").map(_s)
    exp_t = col("apakah sebelumnya memiliki pengalaman").map(_s).str.lower()
    exp_f = col("apakah pernah bekerja di perusahaan bidang").map(_s).str.lower()

    def lvl(v):
        return 0.0 if v.startswith("tidak") else 2.0 if "lebih dari" in v else 1.0 if "kurang dari" in v else np.nan

    out = pd.DataFrame(index=raw.index)
    out["era"] = "v2"
    out["ts"], out["nama"], out["hp"], out["ktp"] = ts, col("nama lengkap"), col("nomor handphone"), col("nik")
    out["usia"], out["pend_level"] = usia, pend.map(_pend)
    out["fresh_grad"] = pend.str.lower().str.contains("freshgraduate|fresh graduate|tanpa pengalaman").astype(float)
    out["exp_tele_ord"], out["exp_finance_ord"] = exp_t.map(lvl), exp_f.map(lvl)
    out["exp_tele"], out["exp_finance"] = (out["exp_tele_ord"] > 0).astype(float), (out["exp_finance_ord"] > 0).astype(float)
    out["exp_sales"] = np.nan
    br = col("jarak tempat tinggal").map(jarak_bracket)
    rows = []
    for alamat, b in zip(col("alamat lengkap"), br):
        km, lv, cf, mt = geo.km(alamat=_s(alamat), alamat_only=True)
        rows.append(_geo_cols(km, lv, cf, mt, b))
    g = pd.DataFrame(rows, index=raw.index, columns=["jarak_km", "jarak_src", "geo_conf", "geo_match", "geo_level"])
    out = out.join(g)
    out["ext_jarak_bracket_km"] = br
    out["ext_ms_office"] = col("apakah terbiasa mengoperasikan").map(_s).map(
        lambda v: 2.0 if v.lower().startswith("ya") else 1.0 if v.lower().startswith("jarang") else 0.0 if v else np.nan)
    out["ext_siap_join"] = col("apakah saat ini available").map(_s).str.lower().str.startswith("ya").astype(float)
    return out


def adapt_v3(raw: pd.DataFrame, geo: Geo) -> pd.DataFrame:
    c = {re.sub(r"\s+", " ", str(k)).strip().lower(): k for k in raw.columns}

    def col(prefix, nth=0):
        ks = [v for kk, v in c.items() if kk.startswith(prefix)]
        return raw[ks[nth]]

    raw = raw[col("posisi yang dilamar").map(_s).str.lower().str.contains("tele marketing")]
    c = {re.sub(r"\s+", " ", str(k)).strip().lower(): k for k in raw.columns}
    ts = pd.to_datetime(col("waktu mulai"), errors="coerce")
    dob = col("tanggal lahir").map(parse_dob)
    j1, p1, t1 = col("jabatan pekerjaan terakhir"), col("nama perusahaan bekerja terakhir"), col("periode kerja")
    j2, p2, t2 = col("sebelum pekerjaan terakhir"), col("perusahaan tempat anda bekerja"), col("sebutkan tanggal anda mulai")
    rows = []
    for a, b, d, e, f_, g_, tt in zip(j1, p1, t1, j2, p2, t2, ts):
        ref = tt if pd.notna(tt) else pd.Timestamp.today()
        jobs = []
        for jab, per, per_t in ((a, b, d), (e, f_, g_)):
            jab_s = _s(jab)
            if jab_s.lower() in NONE_TXT:
                continue
            text = jab_s + " " + _s(per)
            jobs.append((text, months_between(per_t, ref)))
        has_job = bool(jobs)

        def agg(kws):
            m = [mo for tx, mo in jobs if _has(tx, kws)]
            if not m:
                return 0.0, 0.0
            tot = np.nansum(m) if not all(np.isnan(m)) else np.nan
            return 1.0, _ord(tot, True)

        et, eto = agg(TELE_KW)
        ef, efo = agg(FIN_KW)
        es, _ = agg(SALES_KW)
        rows.append((0.0 if has_job else 1.0, et, eto, ef, efo, es))
    r = pd.DataFrame(rows, index=raw.index, columns=["fresh_grad", "exp_tele", "exp_tele_ord", "exp_finance", "exp_finance_ord", "exp_sales"])
    out = pd.DataFrame(index=raw.index)
    out["era"] = "v3"
    out["ts"], out["nama"], out["hp"], out["ktp"] = ts, col("nama lengkap"), col("nomor handphone"), col("nik ktp")
    out["usia"] = [usia_from(d, t) for d, t in zip(dob, ts)]
    out["pend_level"] = col("pendidikan terakhir").map(_pend)
    out = out.join(r)
    gr = []
    for kab in col("domisili"):
        km, lv, cf, mt = geo.km(kab=_s(kab))
        gr.append(_geo_cols(km, lv, cf, mt))
    out = out.join(pd.DataFrame(gr, index=raw.index, columns=["jarak_km", "jarak_src", "geo_conf", "geo_match", "geo_level"]))
    return out


# ------------------------------------------------------------------ rekap & label
def load_rekap_dated(path, unit="tele marketing") -> pd.DataFrame:
    warnings.filterwarnings("ignore")
    x = pd.ExcelFile(path)
    kf = x.parse("Kandidat Fix")
    parts = [pd.DataFrame({"nama": kf["Nama"], "hp": kf["No HP"], "unit": kf["Unit"],
                           "d": pd.to_datetime(kf["Tanggal Interview"], errors="coerce")})]
    if "Kandidat" in x.sheet_names:
        k = x.parse("Kandidat")
        parts.append(pd.DataFrame({"nama": k["Nama Kandidat"], "hp": None, "unit": k["Unit"],
                                   "d": pd.to_datetime(k["Tanggal Kirim"], errors="coerce")}))
    r = pd.concat(parts, ignore_index=True)
    r = r[r["nama"].notna() & r["unit"].astype(str).str.lower().str.contains(unit)]
    r["n"], r["h"] = r["nama"].map(norm_name), r["hp"].map(norm_hp)
    return r[r["n"].str.len() > 2].reset_index(drop=True)


class Matcher:
    def __init__(self, rek: pd.DataFrame):
        self.by_n: dict[str, list] = {}
        self.by_h: dict[str, list] = {}
        for n, h, d in zip(rek["n"], rek["h"], rek["d"]):
            self.by_n.setdefault(n, []).append(d)
            if h:
                self.by_h.setdefault(h, []).append((n, d))

    def dates(self, nama, hp):
        """Daftar tanggal interview pelamar ini di rekap (NaT = ada di rekap tapi tanggal kosong). None = tidak ada."""
        n, h = norm_name(nama), norm_hp(hp)
        if n in self.by_n:
            return list(self.by_n[n])
        if h in self.by_h:
            sc = [(fuzz.token_sort_ratio(n, m), d) for m, d in self.by_h[h]]
            if max(s for s, _ in sc) >= 60:
                return [d for s, d in sc if s >= 60]
        return None


def person_key(ktp, nama, hp) -> str:
    k = re.sub(r"\D", "", str(ktp)) if not pd.isna(ktp) else ""
    return "k" + k if len(k) >= 10 else "h" + norm_hp(hp) + "|" + norm_name(nama)[:12]


def assign_labels(df: pd.DataFrame, rek: pd.DataFrame, max_lag_days: int = 90, censor_days: int = 14) -> pd.DataFrame:
    """Satu baris per orang. Label 1 bila ada interview di rekap pada [submit-1 hari, submit+90 hari]. Bila rekap HANYA berisi
    interview sebelum submit (melamar ulang) -> label NaN (tidak jelas). Pilih submit terakhir sebelum tanggal interview."""
    M = Matcher(rek)
    cutoff = rek["d"].max() - pd.Timedelta(days=censor_days)  # submit setelah ini belum sempat diproses -> label belum diketahui
    df = df.copy().reset_index(drop=True)
    df["person"] = [person_key(k, n, h) for k, n, h in zip(df["ktp"], df["nama"], df["hp"])]
    df["_d"] = [M.dates(n, h) for n, h in zip(df["nama"], df["hp"])]
    out = []
    for _p, g in df.sort_values("ts").groupby("person", sort=False):
        ds = g["_d"].iloc[-1]
        if ds is None:
            r = g.iloc[-1].copy()
            if r["ts"] > cutoff:
                r["y"], r["label_note"], r["rekap_date"] = np.nan, f"belum diketahui (rekap berakhir {rek['d'].max():%d-%m-%Y}, submit terlalu baru)", pd.NaT
            else:
                r["y"], r["label_note"], r["rekap_date"] = 0.0, "tidak ada di rekap", pd.NaT
            out.append(r)
            continue
        known = [d for d in ds if pd.notna(d)]
        if not known:  # ada di rekap tapi tanggal interview kosong
            r = g.iloc[-1].copy()
            r["y"], r["label_note"], r["rekap_date"] = 1.0, "ada di rekap (tanggal kosong)", pd.NaT
            out.append(r)
            continue
        chosen = None
        for _, row in g.iterrows():  # submit terakhir yang masih sebelum / dekat tanggal interview
            ok = [d for d in known if row["ts"].normalize() - pd.Timedelta(days=1) <= d <= row["ts"].normalize() + pd.Timedelta(days=max_lag_days)]
            if ok:
                chosen = (row.copy(), min(ok))
        if chosen:
            r, d = chosen
            r["y"], r["label_note"], r["rekap_date"] = 1.0, "lolos ke interview", d
        else:
            r = g.iloc[-1].copy()
            r["y"], r["label_note"], r["rekap_date"] = np.nan, "melamar ulang setelah pernah interview (label tidak jelas)", min(known)
        out.append(r)
    res = pd.DataFrame(out).drop(columns=["_d"]).reset_index(drop=True)
    res["jarak_bin"] = pd.cut(res["jarak_km"], [-1, 15, 30, 45, 1e9], labels=["1-15", "16-30", "31-45", ">45"]).astype(object)
    res["lokasi_presisi"] = (res["geo_level"] == "kecamatan").astype(float)  # bracket / kab-kota / tidak terbaca = tidak presisi
    return res


def build_unified(v1: pd.DataFrame, v2: pd.DataFrame, v3: pd.DataFrame, rekap_path, geo: Geo | None = None):
    geo = geo or Geo()
    parts = [adapt_v1(v1, geo), adapt_v2(v2, geo), adapt_v3(v3, geo)]
    allr = pd.concat(parts, ignore_index=True)
    allr = allr[allr["ts"].notna() & allr["nama"].notna()]
    return assign_labels(allr, load_rekap_dated(rekap_path))


def mark_training(u: pd.DataFrame, windows: dict) -> pd.DataFrame:
    """Kolom use_train: label jelas (bukan NaN) dan tanggal submit ada di jendela terpercaya generasi form-nya."""
    ok = pd.Series(False, index=u.index)
    for era, rngs in windows.items():
        for a, b in rngs:
            ok |= (u["era"] == era) & (u["ts"] >= pd.Timestamp(a)) & (u["ts"] < pd.Timestamp(b) + pd.Timedelta(days=1))
    u = u.copy()
    u["use_train"] = ok & u["y"].notna()
    return u
