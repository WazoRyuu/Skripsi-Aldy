"""Eksperimen untuk Bab IV skripsi: membandingkan Logistic Regression, Random Forest, XGBoost sesuai protokol Bab III.

  python eksperimen_bab4.py --excel Data_Latih_Tele_Marketing.xlsx --out hasil_bab4

Protokol (sesuai 3.5.3-3.5.5):
  1. Stratified split 80/20 (random_state=42). Data uji hanya dipakai sekali di akhir.
  2. Preprocessing di dalam Pipeline (imputasi median, scaling, one-hot) -> hanya belajar dari data latih.
  3. Hyperparameter tuning: GridSearchCV 5-fold stratified pada data latih, skor ROC-AUC.
  4. Threshold dipilih dari prediksi out-of-fold data latih (memaksimalkan F1), bukan dari data uji.
  5. Metrik: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, + baseline kelas terbanyak.
  6. Karena data uji kecil (n~80): ditambah nested cross-validation (3x5 fold) dan CI bootstrap untuk AUC.
  7. Validasi temporal: leave-one-month-out. Uji pergeseran pola: latih 2024 -> uji 2026 dan sebaliknya.
  8. Interpretasi: koefisien LR, importance RF, SHAP XGBoost.
Skenario: "2024" (kelompok A, utama) dan "2026" (kelompok B+C, pembanding).
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_unified import REN  # noqa: E402

from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,  # noqa: E402
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

NUM = ["usia", "jarak_km", "exp_tele_ord", "exp_finance_ord"]
BIN = ["fresh_grad", "exp_tele", "exp_finance", "lokasi_presisi"]
CAT = ["pend_level"]
NAMA = {"LR": "Logistic Regression", "RF": "Random Forest", "XGB": "XGBoost"}
SEED = 42


def muat(path, skenario):
    d = pd.read_excel(path, sheet_name="Data_Latih").rename(columns=REN)
    d = d[d["y"].notna() & (d["siap"] == "Ya")].copy()
    k = d["kelompok"].str[0]
    d = d[k == "A"] if skenario == "2024" else d[k.isin(["B", "C"])]
    d["jarak_km"] = d["jarak_km"].clip(upper=100)
    d["bulan"] = pd.to_datetime(d["ts"]).dt.to_period("M").astype(str)
    d["y"] = d["y"].astype(int)
    return d.reset_index(drop=True)


def prep(scale):
    n = [("i", SimpleImputer(strategy="median"))] + ([("s", StandardScaler())] if scale else [])
    return ColumnTransformer([("n", Pipeline(n), NUM), ("b", SimpleImputer(strategy="constant", fill_value=0), BIN),
                              ("c", OneHotEncoder(handle_unknown="ignore"), CAT)])


def model_spec():
    return {
        "LR": (Pipeline([("p", prep(True)), ("m", LogisticRegression(max_iter=2000, random_state=SEED))]),
               {"m__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10], "m__class_weight": [None, "balanced"]}),
        "RF": (Pipeline([("p", prep(False)), ("m", RandomForestClassifier(random_state=SEED, n_jobs=1))]),
               {"m__n_estimators": [200], "m__max_depth": [3, 5, None], "m__min_samples_leaf": [5, 15]}),
        "XGB": (Pipeline([("p", prep(False)), ("m", XGBClassifier(random_state=SEED, n_jobs=1, eval_metric="logloss", verbosity=0))]),
                {"m__n_estimators": [100, 250], "m__max_depth": [2, 3], "m__learning_rate": [0.03, 0.1], "m__min_child_weight": [5]}),
    }


def pilih_threshold(y, p):
    grid = np.linspace(0.2, 0.8, 61)
    return float(grid[int(np.argmax([f1_score(y, p >= t) for t in grid]))])


def metrik(y, p, thr):
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return dict(accuracy=accuracy_score(y, yh), precision=precision_score(y, yh, zero_division=0), recall=recall_score(y, yh),
                f1=f1_score(y, yh), auc=roc_auc_score(y, p), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), threshold=thr)


def boot_auc(y, p, n=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) == 2:
            v.append(roc_auc_score(y[i], p[i]))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def jalankan(d, nama_skenario, out, detail=True):
    X, y = d[NUM + BIN + CAT], d["y"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    hasil, terbaik, proba_te, fitted = {}, {}, {}, {}
    base_acc = max(yte.mean(), 1 - yte.mean())
    for k, (pipe, grid) in model_spec().items():
        gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=1, refit=True).fit(Xtr, ytr)
        oof = cross_val_predict(gs.best_estimator_, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
        thr = pilih_threshold(ytr, oof)
        p = gs.best_estimator_.predict_proba(Xte)[:, 1]
        m = metrik(yte, p, thr)
        m.update(cv_auc_latih=float(gs.best_score_), auc_ci=boot_auc(yte, p), params={a.replace("m__", ""): b for a, b in gs.best_params_.items()})
        hasil[k], proba_te[k], fitted[k], terbaik[k] = m, p, gs.best_estimator_, gs.best_params_
    # nested CV (robust terhadap data uji kecil)
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
    nest = {k: [] for k in hasil}
    for tr, te in outer.split(X, y):
        for k, (pipe, grid) in model_spec().items():
            inner = GridSearchCV(pipe, grid, scoring="roc_auc", cv=StratifiedKFold(3, shuffle=True, random_state=SEED), n_jobs=1).fit(X.iloc[tr], y.iloc[tr])
            nest[k].append(roc_auc_score(y.iloc[te], inner.predict_proba(X.iloc[te])[:, 1]))
    for k in hasil:
        hasil[k]["nested_auc_mean"], hasil[k]["nested_auc_std"] = float(np.mean(nest[k])), float(np.std(nest[k]))
    ringkas = dict(skenario=nama_skenario, n=len(d), n_latih=len(Xtr), n_uji=len(Xte), pos_rate=float(y.mean()), pos_rate_uji=float(yte.mean()),
                   baseline_acc_uji=float(base_acc), hasil=hasil)
    if not detail:
        return ringkas, None
    # validasi temporal: leave-one-month-out dgn parameter terbaik
    lomo = {k: [] for k in hasil}
    bln = d["bulan"]
    for b in sorted(bln.unique()):
        te = (bln == b).values
        if te.sum() < 15 or y[te].nunique() < 2 or y[~te].nunique() < 2:
            continue
        for k, (pipe, _) in model_spec().items():
            mdl = pipe.set_params(**terbaik[k]).fit(X[~te], y[~te])
            lomo[k].append((b, int(te.sum()), float(roc_auc_score(y[te], mdl.predict_proba(X[te])[:, 1]))))
    ringkas["lomo"] = {k: dict(bulan=v, rata2=float(np.average([a[2] for a in v], weights=[a[1] for a in v])), min=float(min(a[2] for a in v)),
                               max=float(max(a[2] for a in v))) for k, v in lomo.items() if v}
    return ringkas, dict(Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte, proba=proba_te, fitted=fitted, X=X, y=y)


def pergeseran(d24, d26):
    """Latih di satu era, uji di era lain (dengan parameter hasil tuning pada era latih)."""
    out = {}
    for nm, (a, b) in {"2024->2026": (d24, d26), "2026->2024": (d26, d24)}.items():
        out[nm] = {}
        for k, (pipe, grid) in model_spec().items():
            gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=1).fit(a[NUM + BIN + CAT], a["y"])
            p = gs.best_estimator_.predict_proba(b[NUM + BIN + CAT])[:, 1]
            out[nm][k] = dict(auc=float(roc_auc_score(b["y"], p)), ci=boot_auc(b["y"], p), cv_auc_dalam_era=float(gs.best_score_))
    return out


def gambar(S, ex, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    surf, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
    col = {"LR": "#2a78d6", "RF": "#eb6834", "XGB": "#1baf7a"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": grid, "axes.labelcolor": ink2, "xtick.color": ink2, "ytick.color": ink2,
                         "figure.facecolor": surf, "axes.facecolor": surf, "text.color": ink})
    yte = ex["yte"].values
    # ROC
    fig, ax = plt.subplots(figsize=(5.6, 5), dpi=170)
    ax.plot([0, 1], [0, 1], color="#c9c8c3", lw=1.2, ls="--")
    for k in NAMA:
        f, t, _ = roc_curve(yte, ex["proba"][k])
        ax.plot(f, t, color=col[k], lw=2, label=f"{NAMA[k]} (AUC {S['hasil'][k]['auc']:.2f})".replace(".", ","))
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.grid(color=grid, lw=.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="lower right", fontsize=9); fig.tight_layout(); fig.savefig(out / "roc_2024.png"); plt.close(fig)
    # Confusion matrix
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.4), dpi=170)
    for ax, k in zip(axs, NAMA):
        h = S["hasil"][k]; cm = np.array([[h["tn"], h["fp"]], [h["fn"], h["tp"]]])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.15)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center", fontsize=15, fontweight="bold", color="white" if v > cm.max() * .55 else ink)
        ax.set_xticks([0, 1], ["Tidak lanjut", "Lanjut"]); ax.set_yticks([0, 1], ["Tidak lanjut", "Lanjut"])
        ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual"); ax.set_title(NAMA[k], fontsize=11, color=ink)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout(); fig.savefig(out / "confusion_2024.png"); plt.close(fig)
    return col, ink, ink2, grid, surf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", required=True)
    ap.add_argument("--out", default="hasil_bab4")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    d24, d26 = muat(a.excel, "2024"), muat(a.excel, "2026")
    S24, ex = jalankan(d24, "2024", out)
    S26, _ = jalankan(d26, "2026", out, detail=False)
    pg = pergeseran(d24, d26)
    gambar(S24, ex, out)
    # SHAP (XGBoost) + importance RF + koefisien LR, pada seluruh data 2024 dengan model yang dilatih pada data latih
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap
    fx = ex["fitted"]["XGB"]
    Xall = ex["X"]; Z = fx.named_steps["p"].transform(Xall); nm = [n.split("__", 1)[1] for n in fx.named_steps["p"].get_feature_names_out()]
    sv = shap.TreeExplainer(fx.named_steps["m"]).shap_values(Z)
    imp = pd.Series(np.abs(sv).mean(0), index=nm).sort_values()
    sgn = pd.Series([np.corrcoef(Z[:, i].astype(float), sv[:, i])[0, 1] if np.std(Z[:, i]) > 0 else 0 for i in range(Z.shape[1])], index=nm)
    fl = ex["fitted"]["LR"]; coef = pd.Series(fl.named_steps["m"].coef_[0], index=[n.split("__", 1)[1] for n in fl.named_steps["p"].get_feature_names_out()])
    rf = ex["fitted"]["RF"]; rfi = pd.Series(rf.named_steps["m"].feature_importances_, index=[n.split("__", 1)[1] for n in rf.named_steps["p"].get_feature_names_out()])
    tab = pd.DataFrame({"SHAP_XGB_mean_abs": imp, "arah_SHAP(+naik)": sgn[imp.index].round(2), "RF_importance": rfi[imp.index], "LR_koefisien": coef[imp.index]}).sort_values("SHAP_XGB_mean_abs", ascending=False)
    tab.to_csv(out / "importance_2024.csv", float_format="%.4f")
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=170)
    s = imp.tail(10)
    ax.barh(s.index, s.values, color=["#2a78d6" if sgn[i] >= 0 else "#e34948" for i in s.index], height=.6)
    for i, v in enumerate(s.values):
        ax.text(v + imp.max() * .02, i, f"{v:.3f}".replace(".", ","), va="center", fontsize=8.5, color="#52514e")
    ax.set_xlabel("Rata-rata |SHAP| (kontribusi terhadap prediksi XGBoost)", fontsize=9); ax.set_xlim(0, imp.max() * 1.18)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title("Biru = nilai tinggi menaikkan peluang · Merah = menurunkan", fontsize=9, color="#52514e", loc="left")
    fig.tight_layout(); fig.savefig(out / "shap_2024.png"); plt.close(fig)
    # perbandingan AUC antar skenario
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=170)
    ks = list(NAMA); w = .26; xs = np.arange(3)
    lab = ["Uji 20%\n(2024)", "Nested CV\n(2024)", "Nested CV\n(2026)", ]
    col = {"LR": "#2a78d6", "RF": "#eb6834", "XGB": "#1baf7a"}
    for j, k in enumerate(ks):
        vals = [S24["hasil"][k]["auc"], S24["hasil"][k]["nested_auc_mean"], S26["hasil"][k]["nested_auc_mean"]]
        ax.bar(xs + (j - 1) * w, vals, w * .92, color=col[k], label=NAMA[k])
        for x, v in zip(xs + (j - 1) * w, vals):
            ax.text(x, v + .008, f"{v:.2f}".replace(".", ","), ha="center", fontsize=8, color="#52514e")
    ax.axhline(.5, color="#8d8c86", lw=1, ls="--"); ax.text(-0.5, .507, "acak = 0,50", fontsize=8, color="#52514e", ha="left")
    ax.set_xticks(xs, lab, fontsize=9); ax.set_ylim(.4, .8); ax.set_ylabel("ROC-AUC", fontsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.12))
    fig.tight_layout(); fig.savefig(out / "auc_skenario.png"); plt.close(fig)
    # distribusi label per bulan (2024 & 2026)
    allm = pd.concat([d24.assign(era="2024"), d26.assign(era="2026")])
    g = allm.groupby("bulan")["y"].agg(["mean", "count"]); g.to_csv(out / "label_per_bulan.csv")
    json.dump(dict(S24=S24, S26=S26, pergeseran=pg), open(out / "hasil.json", "w"), indent=1, default=str)
    print(json.dumps(dict(S24=S24, S26=S26, pergeseran=pg), indent=1, default=str)[:6000])
    print(tab.round(3).to_string())


if __name__ == "__main__":
    main()
