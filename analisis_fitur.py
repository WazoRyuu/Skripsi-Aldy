"""Faktor apa yang paling berhubungan dengan 'lanjut interview'? Korelasi, matriks korelasi, dan peringkat fitur.

  python analisis_fitur.py --excel Data_Latih_Tele_Marketing.xlsx --out analisis
  python analisis_fitur.py --excel ... --v1 Form_Loker.xlsx --rekap Rekap_All.xlsx   # tambah analisis fitur LENGKAP form lama (2024)

Untuk tiap skenario (2024 / 2026 / semua) dihasilkan:
  korelasi_<skenario>.png   matriks korelasi antar fitur + label
  peringkat_<skenario>.png  fitur paling berhubungan dengan label (korelasi, tanda bintang = signifikan setelah koreksi)
  peringkat_fitur.xlsx      tabel lengkap: korelasi, p-value, q (BH), AUC tunggal, mutual information, KONSISTENSI antar bulan
Ukuran kunci adalah konsistensi antar bulan: fitur yang hanya kuat di satu-dua bulan biasanya jejak batch, bukan pola nyata.
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

NAMA = {"usia": "Usia", "fresh_grad": "Fresh graduate", "exp_tele": "Pernah tele/CS/collection", "exp_tele_ord": "Lama pengalaman tele",
        "exp_finance": "Pernah multifinance/bank", "exp_finance_ord": "Lama pengalaman finance", "jarak_km": "Jarak ke Cisauk (km)",
        "lokasi_presisi": "Domisili terbaca presisi", "ext_menikah": "Menikah*", "ext_punya_anak": "Punya anak*", "ext_nilai_norm": "Nilai/IPK*",
        "ext_kuliah_sekarang": "Sedang kuliah*", "ext_ms_office": "Mahir MS Office**", "ext_siap_join": "Siap segera join**",
        "ext_jarak_bracket_km": "Jarak pilihan pelamar**", "exp_sales": "Pengalaman sales*"}
CATATAN = "* hanya form lama   ** hanya form kuis"
BG, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
POS, NEG, MID = "#2a78d6", "#e34948", "#f0efec"  # diverging biru <-> merah, titik tengah abu


def design(df: pd.DataFrame, min_nonnull=0.3) -> pd.DataFrame:
    cols = [c for c in NAMA if c in df.columns and df[c].notna().mean() >= min_nonnull and df[c].nunique() > 1]
    X = df[cols].astype(float).copy()
    for lv in ("S1", "SMA/SMK", "D3"):
        if "pend_level" in df.columns and (df["pend_level"] == lv).sum() >= 15:
            X[f"pend_{lv}"] = (df["pend_level"] == lv).astype(float)
            NAMA[f"pend_{lv}"] = f"Pendidikan {lv}"
    return X


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = np.empty_like(p)
    q[o] = np.minimum.accumulate((p[o] * len(p) / (np.arange(len(p)) + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def ranking(X: pd.DataFrame, y: pd.Series, bulan: pd.Series) -> pd.DataFrame:
    from scipy.stats import pearsonr
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.metrics import roc_auc_score

    rows = []
    Xf = X.fillna(X.median())
    mi = mutual_info_classif(Xf, y, discrete_features=[c.startswith("pend_") or X[c].nunique() <= 3 for c in X.columns], random_state=0)
    for c, m in zip(X.columns, mi):
        ok = X[c].notna()
        r, p = pearsonr(X.loc[ok, c], y[ok])
        auc = roc_auc_score(y[ok], X.loc[ok, c])
        per = []
        for _b, idx in bulan.groupby(bulan).groups.items():
            ii = [i for i in idx if ok[i]]
            if len(ii) >= 25 and y[ii].nunique() == 2 and X.loc[ii, c].nunique() > 1:
                per.append(np.corrcoef(X.loc[ii, c], y[ii])[0, 1])
        same = int(sum(np.sign(v) == np.sign(r) for v in per)) if per else 0
        rows.append(dict(fitur=NAMA.get(c, c), kode=c, n=int(ok.sum()), korelasi=r, p_value=p, auc_tunggal=auc, mutual_info=m,
                         bulan_diuji=len(per), bulan_searah=same))
    d = pd.DataFrame(rows)
    d["q_BH"] = bh(d["p_value"])
    d["konsistensi"] = d.apply(lambda r: f"{r.bulan_searah}/{r.bulan_diuji}", axis=1)
    return d.sort_values("korelasi", key=np.abs, ascending=False).reset_index(drop=True)


def _style():
    import matplotlib as mpl
    mpl.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
                         "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": "#c3c2b7"})


def heatmap(X, y, title, path):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    _style()
    D = X.copy()
    D["LABEL lanjut interview"] = y.values
    names = [NAMA.get(c, c) for c in X.columns] + ["LABEL lanjut interview"]
    C = D.corr().values
    cmap = LinearSegmentedColormap.from_list("div", [NEG, MID, POS])
    fig, ax = plt.subplots(figsize=(0.62 * len(names) + 4, 0.62 * len(names) + 2.4))
    M = np.ma.array(C, mask=np.triu(np.ones_like(C, bool), 0))  # diagonal (selalu 1) disembunyikan
    im = ax.imshow(M, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    for i in range(len(names)):
        for j in range(i):
            v = C[i, j]
            if abs(v) >= 0.15 or i == len(names) - 1:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=INK if abs(v) < 0.6 else "white")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.add_patch(plt.Rectangle((-0.5, len(names) - 1.5), len(names), 1, fill=False, ec=INK, lw=1.2))
    cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label("korelasi (biru = naik bersama, merah = berlawanan)", fontsize=8, color=MUTED)
    cb.outline.set_visible(False)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=12)
    fig.text(0.01, 0.005, "Baris LABEL (kotak hitam) = hubungan tiap fitur dengan lolos ke interview. " + CATATAN, fontsize=7, color=MUTED)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def bars(rank, title, path, top=12):
    import matplotlib.pyplot as plt

    _style()
    r = rank.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.46 * len(r) + 1.6))
    col = [POS if v > 0 else NEG for v in r["korelasi"]]
    ax.barh(range(len(r)), r["korelasi"], color=col, height=0.62)
    ax.axvline(0, color="#c3c2b7", lw=1)
    ax.set_yticks(range(len(r)), r["fitur"], fontsize=9)
    lim = max(0.3, r["korelasi"].abs().max() * 1.2)
    ax.set_xlim(-lim, lim * 2.2)
    ax.set_xticks([t for t in np.arange(-0.3, 0.31, 0.1) if abs(t) <= lim + 1e-9])
    for i, (v, q, k) in enumerate(zip(r["korelasi"], r["q_BH"], r["konsistensi"])):  # label di kolom kanan tetap: tidak menabrak nama fitur
        sig = "signifikan" if q < 0.05 else "tidak signifikan"
        ax.text(lim * 1.06, i, f"{v:+.2f}  {sig}  ·  searah {k} bulan", va="center", ha="left", fontsize=8.5, color=INK)
    ax.set_xlabel("korelasi dengan lanjut interview (kanan = menaikkan peluang)", fontsize=8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(left=False)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    fig.text(0.01, -0.01, "'signifikan' = lolos koreksi Benjamini-Hochberg (q<0,05).  'searah k/n bulan' = di berapa bulan arah hubungannya sama.  " + CATATAN,
             fontsize=7, color=MUTED)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def scenario_frames(d):
    k = d["kelompok"].str[0]
    return {"2024": d[k == "A"], "2026": d[k.isin(["B", "C"])], "semua": d[k.isin(["A", "B", "C"])]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", required=True)
    ap.add_argument("--out", default="analisis")
    ap.add_argument("--v1")
    ap.add_argument("--rekap")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = pd.read_excel(a.excel, sheet_name="Data_Latih")
    ren = {"Usia": "usia", "Pendidikan": "pend_level", "Fresh Graduate": "fresh_grad", "Pengalaman Tele (ada)": "exp_tele",
           "Pengalaman Tele (lama)": "exp_tele_ord", "Pengalaman Finance (ada)": "exp_finance", "Pengalaman Finance (lama)": "exp_finance_ord",
           "Jarak ke Cisauk (km)": "jarak_km", "Lokasi Presisi": "lokasi_presisi", "LABEL (1=lanjut interview)": "y", "Siap Dipakai Training?": "siap",
           "Kelompok Data": "kelompok", "Tanggal Submit": "ts", "Pengalaman Sales": "exp_sales", "[Kuis] Jarak Pilihan Pelamar (km)": "ext_jarak_bracket_km",
           "[Kuis] MS Office": "ext_ms_office", "[Kuis] Siap Join": "ext_siap_join", "[Lama] Menikah": "ext_menikah", "[Lama] Punya Anak": "ext_punya_anak",
           "[Lama] Nilai (0-100)": "ext_nilai_norm", "[Lama] Sedang Kuliah": "ext_kuliah_sekarang"}
    d = d.rename(columns=ren)
    d = d[d["y"].notna() & (d["siap"] == "Ya")].copy()
    d["jarak_km"] = d["jarak_km"].clip(upper=100)
    d["bulan"] = pd.to_datetime(d["ts"]).dt.to_period("M").astype(str)
    allr = []
    for nm, f in scenario_frames(d).items():
        f = f.reset_index(drop=True)
        X, y = design(f), f["y"].astype(int)
        rk = ranking(X, y, f["bulan"])
        rk.insert(0, "skenario", nm)
        allr.append(rk)
        label = {"2024": "Data 2024 (form lama)", "2026": "Data 2026 (Des 2025 - Agu 2026)", "semua": "Semua data"}[nm]
        heatmap(X, y, f"Matriks korelasi — {label} (n={len(f)})", out / f"korelasi_{nm}.png")
        bars(rk, f"Faktor paling berhubungan dengan lanjut interview — {label} (n={len(f)})", out / f"peringkat_{nm}.png")
        print(f"\n== {label}  n={len(f)}  lanjut={y.mean():.0%}")
        print(rk.head(8)[["fitur", "korelasi", "q_BH", "auc_tunggal", "konsistensi"]].round(3).to_string(index=False))
    with pd.ExcelWriter(out / "peringkat_fitur.xlsx") as xw:
        pd.concat(allr).round(4).to_excel(xw, sheet_name="Peringkat", index=False)
    if a.v1 and a.rekap:  # fitur LENGKAP form lama (bidang, bagian, level, nilai, status, dll) hanya untuk 2024
        from config import DEFAULT_PLACEMENT, RELIABLE_WINDOWS
        from features import build_features, dedupe_applicants
        from labeling import relabel
        from modeling import prepare_X

        raw = pd.read_excel(a.v1)
        raw["Lanjut Interview"] = relabel(raw, a.rekap, units=("tele marketing",)).astype(bool)
        feat = dedupe_applicants(build_features(raw, None, DEFAULT_PLACEMENT))
        ts = feat["_ts"]
        m = pd.Series(False, index=feat.index)
        for s, e in RELIABLE_WINDOWS:
            m |= (ts >= pd.Timestamp(s)) & (ts <= pd.Timestamp(e) + pd.Timedelta(days=1))
        rel = feat[m].reset_index(drop=True)
        Xr = prepare_X(rel)
        Xd = pd.get_dummies(Xr, columns=list(Xr.select_dtypes(exclude="number").columns), dtype=float)
        Xd = Xd.loc[:, Xd.sum() >= 15]
        Xd = Xd.loc[:, Xd.nunique() > 1]
        for c in Xd.columns:
            NAMA.setdefault(c, c.replace("_", " "))
        rk = ranking(Xd, rel["_y"].astype(int), pd.to_datetime(rel["_ts"]).dt.to_period("M").astype(str))
        rk.to_csv(out / "peringkat_fitur_lengkap_2024.csv", index=False)
        bars(rk, f"Fitur LENGKAP form lama — 2024 (n={len(rel)})", out / "peringkat_lengkap_2024.png", top=15)
        print("\n== fitur lengkap 2024\n", rk.head(12)[["fitur", "korelasi", "q_BH", "konsistensi"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
