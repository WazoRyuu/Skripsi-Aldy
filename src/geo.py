"""Resolusi domisili (teks bebas dari Google Form) -> koordinat kecamatan -> jarak ke lokasi penempatan.

Sumber koordinat: centroid polygon kecamatan (cahyadsn/wilayah_boundaries, Kepmendagri 2025).
Jarak = garis lurus (haversine). Jarak tempuh jalan biasanya ~1.3-1.6x lebih panjang.
"""
from __future__ import annotations

import math
import re
import unicodedata
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

DATA = Path(__file__).resolve().parent.parent / "data"

# Titik lokasi penempatan. None -> pakai centroid kecamatan bernama sama (mis. "Cisauk").
# Kalau punya titik kantor yang presisi, isi (lat, lng), contoh: {"cisauk": (-6.3393, 106.6302)}
PLACEMENTS: dict[str, tuple[float, float] | None] = {
    "cisauk": None,
    "bsd": (-6.3017, 106.6530), "bsd city": (-6.3017, 106.6530),  # pusat BSD City (perkiraan); ganti dgn titik kantor bila ada
}

STOP = re.compile(
    r"\b(kabupaten|kab|kota|kotamadya|kecamatan|kec|kelurahan|kel|desa|provinsi|prov|"
    r"daerah khusus ibukota|dki|rt|rw|jl|jln|jalan|no|blok|perumahan|perum|komplek|kompleks|"
    r"kp|kampung|dusun|administrasi)\b"
)

ALIAS = {  # singkatan / salah ketik umum -> nama baku
    "tangsel": "tangerang selatan", "tanggerang": "tangerang", "tangerng": "tangerang",
    "tengerang": "tangerang", "jaksel": "jakarta selatan", "jakbar": "jakarta barat",
    "jakpus": "jakarta pusat", "jaktim": "jakarta timur", "jakut": "jakarta utara",
    "baten": "banten", "bante": "banten", "banyen": "banten", "banteb": "banten",
    "jabar": "jawa barat", "jawabarat": "jawa barat",
}


def norm(s) -> str:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return " ".join(ALIAS.get(w, w) for w in s.split())


def strip_admin(s: str) -> str:
    return re.sub(r"\s+", " ", STOP.sub(" ", s)).strip()


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


class GeoResolver:
    def __init__(self):
        kec = pd.read_csv(DATA / "kec_coords.csv", dtype={"kode": str})
        reg = pd.read_csv(DATA / "regions_l12.csv", dtype={"kode": str})
        kec["reg"] = kec["kode"].str[:5]
        kec["prov"] = kec["kode"].str[:2]
        kec["name_n"] = kec["nama"].map(lambda x: strip_admin(norm(x)))
        names = dict(zip(reg["kode"], reg["nama"]))
        kec["reg_name"] = kec["reg"].map(names)
        self.kec = kec.reset_index(drop=True)
        self._kec_names = self.kec["name_n"].tolist()
        self._kec_reg = self.kec["reg"].tolist()
        self._kec_prov = self.kec["prov"].tolist()
        self._kec_rows = self.kec.to_dict("records")

        regs = reg[reg["kode"].str.len() == 5].copy()
        regs["is_kota"] = regs["nama"].str.lower().str.startswith("kota")
        regs["name_n"] = regs["nama"].map(lambda x: strip_admin(norm(x)))
        regs = regs.join(kec.groupby("reg")[["lat", "lng"]].mean(), on="kode").dropna(subset=["lat"])
        self.regs = regs.reset_index(drop=True)
        self._reg_rows = list(zip(self.regs["kode"], self.regs["name_n"], self.regs["is_kota"]))
        self._reg_by_code = {r["kode"]: r for r in self.regs.to_dict("records")}
        provs = reg[reg["kode"].str.len() == 2]
        self._prov_rows = [(k, strip_admin(norm(n))) for k, n in zip(provs["kode"], provs["nama"])]
        self._admin_names = {n for _k, n, _c in self._reg_rows} | {n for _k, n in self._prov_rows}
        self._cache_reg: dict = {}
        self._cache_res: dict = {}

    # ---------- kab/kota ----------
    def _regency_candidates(self, text_n: str, kota_hint):
        key = (text_n, kota_hint)
        if key in self._cache_reg:
            return self._cache_reg[key]
        t = strip_admin(text_n)
        out: dict[str, float] = {}
        if t:
            padded = f" {t} "
            for kode, n, is_kota in self._reg_rows:
                if t == n:
                    s = 100.0
                elif f" {n} " in padded:
                    s = 92.0
                else:
                    s = fuzz.ratio(t, n)
                    if s < 88:
                        continue
                if kota_hint is not None and is_kota != kota_hint:
                    s -= 6  # "Tangerang" polos tetap menerima Kota maupun Kabupaten
                out[kode] = max(out.get(kode, 0), s)
            if t in ("jakarta", "jakarta kota"):  # "jakarta" polos -> semua kota di DKI
                for kode, _n, _k in self._reg_rows:
                    if kode.startswith("31"):
                        out[kode] = max(out.get(kode, 0), 80.0)
        self._cache_reg[key] = out
        return out

    # ---------- utama ----------
    def resolve(self, prov="", kab="", kec="", kel="", alamat="", alamat_w=0.4):
        """alamat_w: bobot petunjuk kab/kota yang disebut di alamat. 0.4 (default) bila ada kolom kab/kota terpisah;
        naikkan (mis. 0.9) bila HANYA ada alamat bebas."""
        key = tuple(map(norm, (prov, kab, kec, kel, alamat))) + (alamat_w,)
        if key not in self._cache_res:
            self._cache_res[key] = self._resolve(*key)
        return self._cache_res[key]

    def _resolve(self, prov_n, kab_n, kec_n, kel_n, alm_n, alamat_w=0.4):
        kota_hint = True if re.search(r"\bkota\b", kab_n) else False if re.search(r"\bkab(upaten)?\b", kab_n) else None

        reg_scores: dict[str, float] = {}
        for txt, w, hint in ((kab_n, 1.0, kota_hint), (prov_n, 0.8, None), (kec_n, 0.5, None),
                             (kel_n, 0.4, None), (alm_n, alamat_w, None)):
            for k, s in self._regency_candidates(txt, hint).items():
                reg_scores[k] = max(reg_scores.get(k, 0), s * w)

        prov_codes = set()
        for kode, pn in self._prov_rows:
            for txt in (prov_n, kab_n):
                t = strip_admin(txt)
                if t and (t == pn or (len(t) > 3 and fuzz.ratio(t, pn) >= 88)):
                    prov_codes.add(kode)
        if re.search(r"\bdki\b|jakarta", prov_n + " " + kab_n):
            prov_codes.add("31")

        keep: set[str] = set()
        if reg_scores:
            top = max(reg_scores.values())
            keep = {k for k, s in reg_scores.items() if s >= top - 15 and s >= 60}
        if keep:  # cari di seluruh provinsi terkait; kecamatan di luar kab/kota yang disebut hanya boleh lewat kolom kecamatan
            provs_keep = {k[:2] for k in keep}
            idx = [i for i, p in enumerate(self._kec_prov) if p in provs_keep]
        elif prov_codes:
            idx = [i for i, p in enumerate(self._kec_prov) if p in prov_codes]
        else:
            idx = list(range(len(self._kec_names)))
        strict = not keep  # tanpa petunjuk kab/kota -> wajib cocok kuat
        names = [self._kec_names[i] for i in idx]

        best = None  # (skor, row, field)
        for field_name, txt, w in (("kecamatan", kec_n, 1.0), ("kelurahan", kel_n, 0.9), ("alamat", alm_n, 0.85),
                                   ("kabupaten", kab_n, 0.8), ("provinsi", prov_n, 0.8)):
            t = strip_admin(txt)
            if len(t) < 3:
                continue
            padded = f" {t} "
            cand: dict[int, float] = {}
            skip_admin = field_name != "kecamatan"  # nama kab/kota di kolom lain bukan nama kecamatan
            for j, n in enumerate(names):
                if len(n) < 3 or (skip_admin and n in self._admin_names):
                    continue
                if t == n:
                    cand[j] = 100.0
                elif len(n) >= 5 and f" {n} " in padded:
                    cand[j] = 93.0
            if field_name in ("kecamatan", "kelurahan"):
                cut = 80 if (keep and field_name == "kecamatan") else 86  # pool kecil (1 kab/kota) -> boleh lebih longgar
                found = process.extract(t, names, scorer=fuzz.ratio, score_cutoff=cut, limit=8)
                # kecocokan longgar (<86) hanya dipakai bila jelas unggul dari kandidat kedua; kalau ambigu (mis. "Cisouk"
                # mirip Cisauk DAN Cisoka) jangan menebak -> jatuh ke pusat kab/kota + catatan di AI_Catatan_Data
                if found and found[0][1] < 86 and len(found) > 1 and found[0][1] - found[1][1] < 5:
                    found = []
                for _n, sc, j in found:
                    if not (skip_admin and names[j] in self._admin_names):
                        cand[j] = max(cand.get(j, 0), sc)
            for j, sc in cand.items():
                r = self._kec_rows[idx[j]]
                outside = bool(keep) and r["reg"] not in keep
                if outside and not (field_name == "kecamatan" and sc >= 93):
                    continue
                tot = sc * w + reg_scores.get(r["reg"], 0) * 0.05 - (10 if outside else 0)
                if field_name == "kecamatan" and norm(r["nama"]) == txt:  # nama persis (mis. "Bangkinang" vs "Bangkinang Kota")
                    tot += 0.5
                if best is None or tot > best[0]:
                    best = (tot, r, field_name, sc)

        if best and (best[3] >= 93 if strict else best[0] >= 74):
            r = best[1]
            return dict(lat=r["lat"], lng=r["lng"], level="kecamatan",
                        match=f"{r['nama']}, {r['reg_name']}", conf=round(min(best[0], 100) / 100, 2))
        if reg_scores:
            k = max(reg_scores, key=reg_scores.get)
            if reg_scores[k] >= 80:
                r = self._reg_by_code[k]
                return dict(lat=r["lat"], lng=r["lng"], level="kabupaten_kota", match=r["nama"],
                            conf=round(reg_scores[k] / 100, 2))
        return dict(lat=float("nan"), lng=float("nan"), level="tidak_terbaca", match="", conf=0.0)

    # ---------- penempatan & jarak ----------
    def placement_coord(self, name: str):
        key = norm(name)
        if PLACEMENTS.get(key):
            return PLACEMENTS[key]
        t0 = strip_admin(key)
        near = [(sc, not self._reg_by_code[k]["is_kota"], k) for k, sc in self._regency_candidates(key, None).items()
                if sc >= 100 and k[:2] in ("31", "32", "36")]  # "Depok"/"Bekasi" = kota di Jabodetabek, bukan kecamatan Depok di Cirebon
        if t0 and near:
            near.sort(key=lambda x: (-x[0], x[1]))
            reg = self._reg_by_code[near[0][2]]
            return float(reg["lat"]), float(reg["lng"])
        r = [row for row in self._kec_rows if row["name_n"] == strip_admin(key)]
        if r:
            pri = {"36": 0, "32": 1, "31": 2}
            r.sort(key=lambda x: pri.get(x["prov"], 9))
            return float(r[0]["lat"]), float(r[0]["lng"])
        # bukan nama kecamatan -> coba sebagai kab/kota ("Bekasi", "Jakarta Selatan"): pakai pusat kab/kota (kota diutamakan)
        t = strip_admin(key)
        cands = [(sc, not self._reg_by_code[k]["is_kota"], k) for k, sc in self._regency_candidates(key, None).items() if sc >= 92]
        if t and cands:
            cands.sort(key=lambda x: (-x[0], x[1]))
            reg = self._reg_by_code[cands[0][2]]
            return float(reg["lat"]), float(reg["lng"])
        raise ValueError(f"Lokasi penempatan '{name}' tidak ditemukan. Tambahkan koordinat di PLACEMENTS.")

    def distance_km(self, lat, lng, placement: str):
        if lat is None or lng is None or (isinstance(lat, float) and math.isnan(lat)):
            return float("nan")
        try:
            plat, plng = self.placement_coord(placement)
        except ValueError:  # penempatan tak dikenal -> jangan hentikan seluruh batch; ditandai di AI_Catatan_Data
            return float("nan")
        return haversine_km(lat, lng, plat, plng)
