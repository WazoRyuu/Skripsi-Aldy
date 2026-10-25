"""Training dari Excel gabungan (Data_Latih_Tele_Marketing.xlsx) + laporan akurasi yang JUJUR (dibanding tebakan paling sederhana).

  python train_unified.py --excel Data_Latih_Tele_Marketing.xlsx --skenario 2024     # kelompok A saja
  python train_unified.py --excel Data_Latih_Tele_Marketing.xlsx --skenario 2026     # kelompok B + C
  python train_unified.py --excel Data_Latih_Tele_Marketing.xlsx --skenario semua    # A + B + C
Opsi: --hati-hati  ikut menyertakan baris "Hati-hati" (mis. kelompok D).  --simpan models/model_unified.joblib  simpan model.

Model ini memakai FITUR INTI yang ada di semua generasi form. Belum dipasang ke Sheet/Streamlit: pasang hanya bila hasilnya terbukti
lebih baik dari baseline pada skenario yang sama.
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
warnings.filterwarnings("ignore")
from build_training_data import BIN, CAT, NUM, _pipe  # noqa: E402

REN = {"Usia": "usia", "Pendidikan": "pend_level", "Fresh Graduate": "fresh_grad", "Pengalaman Tele (ada)": "exp_tele",
       "Pengalaman Tele (lama)": "exp_tele_ord", "Pengalaman Finance (ada)": "exp_finance", "Pengalaman Finance (lama)": "exp_finance_ord",
       "Jarak ke Cisauk (km)": "jarak_km", "Lokasi Presisi": "lokasi_presisi", "LABEL (1=lanjut interview)": "y",
       "Siap Dipakai Training?": "siap", "Kelompok Data": "kelompok", "Tanggal Submit": "ts"}


def load(path, skenario, hati_hati):
    d = pd.read_excel(path, sheet_name="Data_Latih").rename(columns=REN)
    d = d[d["y"].notna() & d["siap"].isin(["Ya", "Hati-hati"] if hati_hati else ["Ya"])].copy()
    k = d["kelompok"].str[0]
    d = d[{"2024": k == "A", "2026": k.isin(["B", "C"]), "semua": k.isin(["A", "B", "C"])}[skenario] if skenario != "all" else slice(None)]
    d["jarak_km"] = d["jarak_km"].clip(upper=100)
    d["bulan"] = pd.to_datetime(d["ts"]).dt.to_period("M")
    return d.reset_index(drop=True)


def main():
    from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
    from sklearn.model_selection import RepeatedStratifiedKFold

    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", required=True)
    ap.add_argument("--skenario", choices=["2024", "2026", "semua"], default="2024")
    ap.add_argument("--hati-hati", action="store_true")
    ap.add_argument("--simpan")
    a = ap.parse_args()
    d = load(a.excel, a.skenario, a.hati_hati)
    y = d["y"].astype(int)
    print(f"skenario={a.skenario}  n={len(d)}  lanjut={y.mean():.1%}")
    oof = np.zeros((5, len(d)))
    for rep in range(5):
        for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42 + rep).split(d, y):
            oof[rep, te] = _pipe().fit(d.iloc[tr], y.iloc[tr]).predict_proba(d.iloc[te])[:, 1]
    p = oof.mean(0)
    thr = float(np.linspace(0.2, 0.8, 61)[np.argmax([((p >= t) & (y == 1)).sum() / y.sum() - ((p >= t) & (y == 0)).sum() / (y == 0).sum()
                                                    for t in np.linspace(0.2, 0.8, 61)])])
    acc, base = accuracy_score(y, p >= thr), max(y.mean(), 1 - y.mean())
    print(f"AUC={roc_auc_score(y, p):.3f}  AP={average_precision_score(y, p):.3f}  akurasi={acc:.3f}  "
          f"baseline(tebak kelas terbanyak)={base:.3f}  selisih={acc - base:+.3f}  threshold={thr:.2f}")
    w = []  # AUC dalam bulan yang sama: apakah model bisa MERANKING pelamar di batch yang sama?
    for _, g in d.assign(p=p).groupby("bulan"):
        if g["y"].nunique() == 2 and len(g) >= 15:
            w.append((roc_auc_score(g["y"], g["p"]), len(g)))
    if w:
        print(f"AUC dalam bulan yang sama (rata-rata tertimbang)={np.average([x for x, _ in w], weights=[n for _, n in w]):.3f}")
    print("\nAkurasi bila hanya memakai pelamar yang model YAKIN (cakupan -> akurasi):")
    conf = np.abs(p - thr)
    for q in (1.0, 0.7, 0.5, 0.3):
        m = conf >= np.quantile(conf, 1 - q)
        print(f"  cakupan {q:>4.0%}  n={int(m.sum()):4d}  akurasi={accuracy_score(y[m], p[m] >= thr):.3f}")
    model = _pipe().fit(d, y)
    names = model.named_steps["p"].get_feature_names_out()
    coef = pd.Series(model.named_steps["m"].coef_[0], index=names).sort_values()
    print("\nPengaruh fitur (koefisien, + = naikkan peluang lanjut):\n", coef.round(2).to_string())
    if a.simpan:
        import joblib
        joblib.dump(dict(pipeline=model, threshold=thr, features=NUM + BIN + CAT, skenario=a.skenario), a.simpan)
        print("model disimpan ->", a.simpan)


if __name__ == "__main__":
    main()
