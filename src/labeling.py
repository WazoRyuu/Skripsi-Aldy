"""Label 'tembus ke interview' dari Rekap_All: cocok lewat No HP (utama) atau nama persis.

Kenapa bukan nama saja: nama sering beda ejaan antara form dan rekap ("Nurlarla" vs "Nurlaela"), No HP lebih stabil.
Untuk menghindari HP salah-ketik / nomor dipakai bersama, kecocokan HP wajib disertai kemiripan nama >= 60.
"""
from __future__ import annotations

import re
import warnings

import pandas as pd
from rapidfuzz import fuzz

UNITS = None  # None = semua unit rekap (lolos seleksi berkas ke unit mana pun = tembus); contoh filter: ("tele marketing",)


def norm_hp(x) -> str:
    s = str(x)
    if re.fullmatch(r"\d+\.0", s):
        s = s[:-2]
    d = re.sub(r"\D", "", s)
    if d.startswith("62"):
        d = d[2:]
    d = d.lstrip("0")
    return d if 8 <= len(d) <= 13 else ""


def norm_name(x) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", str(x).lower())).strip()


def load_rekap(path, units=UNITS) -> pd.DataFrame:
    warnings.filterwarnings("ignore")
    x = pd.ExcelFile(path)
    parts = []
    kf = x.parse("Kandidat Fix")
    parts.append(pd.DataFrame({"nama": kf["Nama"], "hp": kf["No HP"], "unit": kf["Unit"]}))
    if "Kandidat" in x.sheet_names:
        kd = x.parse("Kandidat")
        parts.append(pd.DataFrame({"nama": kd["Nama Kandidat"], "hp": None, "unit": kd["Unit"]}))
    rek = pd.concat(parts, ignore_index=True)
    rek = rek[rek["nama"].notna()]
    if units:
        rek = rek[rek["unit"].astype(str).str.strip().str.lower().isin(units)]
    rek["n"] = rek["nama"].map(norm_name)
    rek["hp"] = rek["hp"].map(norm_hp)
    return rek[rek["n"].str.len() > 2]


def relabel(raw: pd.DataFrame, rekap_path, units=UNITS) -> pd.Series:
    """1 jika pelamar ada di rekap (HP cocok + nama mirip, atau nama persis), else 0. Index = index raw."""
    rek = load_rekap(rekap_path, units)
    names = set(rek["n"])
    by_hp: dict[str, list[str]] = {}
    for h, n in zip(rek["hp"], rek["n"]):
        if h:
            by_hp.setdefault(h, []).append(n)
    cols = {re.sub(r"\s+", " ", str(c)).strip().lower(): c for c in raw.columns}
    c_nama = next(c for k, c in cols.items() if k.startswith("nama lengkap"))
    c_hp = next(c for k, c in cols.items() if k.startswith("nomor hp"))
    out = []
    for nama, hp in zip(raw[c_nama], raw[c_hp]):
        n, h = norm_name(nama), norm_hp(hp)
        ok = n in names or (h in by_hp and max(fuzz.token_sort_ratio(n, m) for m in by_hp[h]) >= 60)
        out.append(int(ok))
    return pd.Series(out, index=raw.index, name="label")
