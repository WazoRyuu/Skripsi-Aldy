"""Pipeline model + penjelasan (alasan) berbahasa Indonesia."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import BIN_FEATURES, CAT_FEATURES, FEATURES, NUM_FEATURES

JARAK_CAP_KM = 100.0  # jarak > 100 km dianggap sama (outlier pelamar luar pulau)


def prepare_X(feat: pd.DataFrame) -> pd.DataFrame:
    X = feat[FEATURES].copy()
    X["jarak_km"] = X["jarak_km"].clip(upper=JARAK_CAP_KM)
    for c in CAT_FEATURES:
        X[c] = X[c].astype(str)
    return X


def make_pipeline(name: str, seed: int = 42) -> Pipeline:
    scale = name == "logreg"
    num = [SimpleImputer(strategy="median")] + ([StandardScaler()] if scale else [])
    pre = ColumnTransformer(
        [
            ("num", Pipeline([(f"s{i}", t) for i, t in enumerate(num)]), NUM_FEATURES),
            ("bin", SimpleImputer(strategy="constant", fill_value=0.0), BIN_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=8, sparse_output=False), CAT_FEATURES),
        ],
        verbose_feature_names_out=True,
    )
    if name == "logreg":
        clf = LogisticRegression(C=0.3, max_iter=2000, class_weight="balanced", random_state=seed)
    elif name == "rf":
        clf = RandomForestClassifier(n_estimators=500, min_samples_leaf=5, max_features="sqrt",
                                     class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    elif name == "xgb":
        from xgboost import XGBClassifier  # import lazy: app web tidak butuh xgboost

        clf = XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.04, subsample=0.8, colsample_bytree=0.7,
                            min_child_weight=3, reg_lambda=3.0, eval_metric="logloss", n_jobs=4, random_state=seed)
    else:
        raise ValueError(name)
    return Pipeline([("pre", pre), ("clf", clf)])


# ----------------------------------------------------------------------------------------------
# Penjelasan: SHAP per fitur -> dikelompokkan per tema -> kalimat Indonesia
# ----------------------------------------------------------------------------------------------
GROUPS = {
    "jarak": ["jarak_km", "dekat_10km", "lokasi_presisi"],
    "usia": ["usia"],
    "pendidikan": ["pend_level", "nilai_norm"],
    "sales": ["pernah_sales", "lama_sales_ord", "jabatan_sales_like"],
    "tele": ["jabatan_tele_like", "bagian"],
    "collection": ["jabatan_collection_like"],
    "crew": ["pernah_crew", "lama_crew_ord"],
    "pengalaman": ["lama_kerja_ord", "durasi_terlama_ord", "level", "bidang"],
    "status_kerja": ["terakhir_kerja"],
    "kuliah": ["kuliah_sekarang", "rencana_kuliah"],
    "keluarga": ["menikah", "punya_anak"],
    "sumber": ["sumber"],
}
FEATURE_TO_GROUP = {f: g for g, fs in GROUPS.items() for f in fs}


def _orig_feature(transformed_name: str) -> str:
    n = transformed_name.split("__", 1)[1]
    for f in sorted(FEATURES, key=len, reverse=True):
        if n == f or n.startswith(f + "_"):
            return f
    return n


class Explainer:
    """Kontribusi per fitur. Untuk Logistic Regression: coef * (x - rata-rata training) (identik dgn SHAP linear);
    untuk model pohon: SHAP TreeExplainer. State-nya berupa dict sederhana (bukan objek kustom), jadi artefak model
    bisa di-load dari notebook, Streamlit, atau script mana pun tanpa mengimpor kelas ini saat menyimpan."""

    def __init__(self, pipe: Pipeline, state: dict | None = None, background_X: pd.DataFrame | None = None):
        self.pipe = pipe
        self.names = list(pipe["pre"].get_feature_names_out())
        self.orig = [_orig_feature(n) for n in self.names]
        clf = pipe["clf"]
        self.linear = isinstance(clf, LogisticRegression)
        if self.linear:
            if state is None:
                state = {"kind": "linear", "mu": pipe["pre"].transform(background_X).mean(axis=0)}
            self.mu = np.asarray(state["mu"], dtype=float)
            self.coef = clf.coef_[0]
        else:
            state = {"kind": "tree"}
            import shap  # import lazy: hanya dibutuhkan untuk model pohon

            self.expl = shap.TreeExplainer(clf)
        self.state = state

    def to_state(self) -> dict:
        return {k: (np.asarray(v) if k == "mu" else v) for k, v in self.state.items()}

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Kontribusi per grup tema (log-odds). Positif = mendorong LANJUT."""
        Z = self.pipe["pre"].transform(X)
        if self.linear:
            sv = (np.asarray(Z, dtype=float) - self.mu) * self.coef
        else:
            sv = self.expl.shap_values(Z)
            if isinstance(sv, list):  # RF versi lama
                sv = sv[1]
            sv = np.asarray(sv)
            if sv.ndim == 3:  # RF versi baru: (n, f, classes)
                sv = sv[:, :, 1]
        df = pd.DataFrame(sv, columns=self.orig, index=X.index)
        df = df.T.groupby(level=0).sum().T
        out = pd.DataFrame(index=X.index)
        for g, fs in GROUPS.items():
            out[g] = df[[f for f in fs if f in df.columns]].sum(axis=1)
        return out


JARAK_BINS = [-0.1, 5, 10, 15, 20, 30, 60, 1e9]
JARAK_LABELS = ["≤5 km", "5–10 km", "10–15 km", "15–20 km", "20–30 km", "30–60 km", ">60 km"]


def compute_rates(feat: pd.DataFrame, y: pd.Series, min_n: int = 8) -> dict:
    """Tingkat lolos historis per kategori (untuk kalimat alasan yang berbasis data)."""
    rates = {"_avg": float(y.mean())}
    for c in ("pend_level", "sumber", "terakhir_kerja", "bagian", "bidang"):
        g = pd.DataFrame({"v": feat[c].astype(str), "y": y.values}).groupby("v")["y"].agg(["mean", "size"])
        rates[c] = {k: (float(m), int(n)) for k, (m, n) in g.iterrows() if n >= min_n}
    b = pd.cut(feat["jarak_km"].clip(upper=1e8), JARAK_BINS, labels=JARAK_LABELS)
    g = pd.DataFrame({"v": b.astype(str), "y": y.values}).groupby("v")["y"].agg(["mean", "size"])
    rates["jarak"] = {k: (float(m), int(n)) for k, (m, n) in g.iterrows() if n >= min_n}
    return rates


def _hist(rates, key, value) -> str:
    if not rates or key not in rates or value not in rates[key]:
        return ""
    m, n = rates[key][value]
    return f" (historis {m:.0%} lolos dari {n} pelamar; rata-rata {rates['_avg']:.0%})"


def _jarak_bucket(d):
    return str(pd.cut([min(d, 1e8)], JARAK_BINS, labels=JARAK_LABELS)[0])


def _jarak_txt(r):
    d = r["jarak_km"]
    return "jarak domisili tidak terbaca" if pd.isna(d) else f"±{d:.0f} km"


def _lama(v):
    return {0: "tanpa pengalaman", 1: "<1 tahun", 2: "1–3 tahun", 3: ">3 tahun"}.get(v, "-")


def reason_text(group: str, positive: bool, r: pd.Series, placement: str, rates: dict | None = None) -> str:
    """Satu kalimat alasan untuk tema `group`, sesuai arah pengaruhnya."""
    if group == "jarak":
        d = r["jarak_km"]
        if pd.isna(d):
            return "Domisili tidak terbaca sehingga jarak ke lokasi penempatan tidak bisa dihitung"
        h = _hist(rates, "jarak", _jarak_bucket(d))
        return (f"Domisili dekat dengan lokasi penempatan {placement}, {_jarak_txt(r)}{h}" if positive
                else f"Domisili cukup jauh dari lokasi penempatan {placement}, {_jarak_txt(r)}{h}")
    if group == "usia":
        u = r["usia"]
        return (f"Usia {u:.0f} tahun berada pada rentang yang umumnya lolos" if positive
                else f"Usia {u:.0f} tahun berada di luar rentang yang umumnya lolos") if not pd.isna(u) else "Usia tidak terbaca"
    if group == "pendidikan":
        n = r["nilai_norm"]
        extra = "" if pd.isna(n) else f", nilai ≈{n:.0f}/100"
        h = _hist(rates, "pend_level", r["pend_level"])
        return (f"Pendidikan {r['pend_level']}{extra} sesuai profil yang umumnya lolos{h}" if positive
                else f"Pendidikan {r['pend_level']}{extra} kurang sesuai profil yang umumnya lolos{h}")
    if group == "sales":
        if positive:
            return f"Punya pengalaman sales/marketing ({_lama(r['lama_sales_ord'])})"
        return "Belum/minim pengalaman sales atau marketing" if r["pernah_sales"] != 1 or r["lama_sales_ord"] <= 1 \
            else "Latar belakang sales kurang menonjol dibanding pelamar yang lolos"
    if group == "tele":
        h = _hist(rates, "bagian", r["bagian"])
        return (f"Pengalaman terakhir di bagian {r['bagian']} relevan dengan telemarketing{h}" if positive
                else f"Pengalaman terakhir ({r['bagian']}) kurang relevan dengan telemarketing{h}")
    if group == "collection":
        return "Punya latar belakang penagihan/collection" if positive else "Tidak punya latar belakang penagihan/collection"
    if group == "crew":
        return "Pengalaman crew store dinilai menunjang" if positive else "Pengalaman crew store kurang menunjang untuk posisi ini"
    if group == "pengalaman":
        h = _hist(rates, "bidang", r["bidang"])
        return (f"Riwayat kerja ({r['bidang']}, level {r['level']}) mendukung{h}" if positive
                else f"Riwayat kerja ({r['bidang']}, level {r['level']}) kurang mendukung{h}")
    if group == "status_kerja":
        h = _hist(rates, "terakhir_kerja", r["terakhir_kerja"])
        return (f"Status kerja terakhir ({r['terakhir_kerja']}) mendukung{h}" if positive
                else f"Status kerja terakhir ({r['terakhir_kerja']}) menurunkan peluang{h}")
    if group == "kuliah":
        return "Status/rencana kuliah tidak menjadi hambatan" if positive else "Sedang atau berencana kuliah, berpotensi mengganggu komitmen kerja"
    if group == "keluarga":
        return "Status pernikahan/keluarga sesuai profil yang umumnya lolos" if positive \
            else "Status pernikahan/keluarga kurang sesuai profil yang umumnya lolos"
    if group == "sumber":
        h = _hist(rates, "sumber", r["sumber"])
        return f"Sumber informasi lowongan ({r['sumber']}) cenderung menghasilkan pelamar yang lolos{h}" if positive \
            else f"Sumber informasi lowongan ({r['sumber']}) cenderung kurang menghasilkan pelamar yang lolos{h}"
    return group


def explain_rows(contrib: pd.DataFrame, feat: pd.DataFrame, proba: np.ndarray, threshold: float,
                 placement: str = "Cisauk", top_k: int = 3, min_abs: float = 0.08, rates: dict | None = None) -> list[dict]:
    rows = []
    for i, (idx, c) in enumerate(contrib.iterrows()):
        p = float(proba[i])
        lanjut = p >= threshold
        pos = c[c > min_abs].sort_values(ascending=False)
        neg = c[c < -min_abs].sort_values()
        main, other = (pos, neg) if lanjut else (neg, pos)
        r = feat.loc[idx]
        pr = [reason_text(g, g in pos.index, r, placement, rates) for g in main.index[:top_k]]
        co = [reason_text(g, g in pos.index, r, placement, rates) for g in other.index[:2]]
        if not pr:
            pr = ["Tidak ada faktor tunggal yang dominan; skor berada dekat batas keputusan"]
        borderline = abs(p - threshold) <= 0.07
        rows.append(dict(
            rekomendasi="LANJUT INTERVIEW" if lanjut else "TIDAK LANJUT",
            skor=round(p, 3),
            keyakinan="Borderline (disarankan tinjau manual)" if borderline else ("Tinggi" if abs(p - threshold) > 0.25 else "Sedang"),
            alasan_utama=" | ".join(pr),
            faktor_pertimbangan=" | ".join(co),
        ))
    return rows
