"""Training + evaluasi.  Pakai:  python train.py --data "Form Loker Tele Marketing (Jawaban) (1).xlsx" """
import argparse, json, sys, warnings
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from config import *
from features import FEATURES, build_features, dedupe_applicants
from modeling import Explainer, compute_rates, make_pipeline, prepare_X
from geo import GeoResolver
from labeling import relabel

warnings.filterwarnings("ignore")


def lib_versions():
    import importlib.metadata as md
    return {k: md.version(k) for k in ("scikit-learn", "xgboost", "shap", "pandas", "numpy", "rapidfuzz", "joblib")}


def in_windows(ts, windows):
    m = pd.Series(False, index=ts.index)
    for a, b in windows:
        m |= (ts >= pd.Timestamp(a)) & (ts <= pd.Timestamp(b) + pd.Timedelta(days=1))
    return m


def best_threshold(y, p):
    # Youden's J (TPR - FPR): lebih stabil daripada F1 saat prevalensi ~40-50% dan kurva F1 datar
    ts = np.linspace(0.05, 0.95, 91)
    y = np.asarray(y)
    j = [((p >= t) & (y == 1)).sum() / max((y == 1).sum(), 1) - ((p >= t) & (y == 0)).sum() / max((y == 0).sum(), 1) for t in ts]
    return float(ts[int(np.argmax(j))])


def metrics(y, p, thr):
    return dict(auc=roc_auc_score(y, p), ap=average_precision_score(y, p), f1=f1_score(y, p >= thr),
                precision=precision_score(y, p >= thr, zero_division=0), recall=recall_score(y, p >= thr), brier=brier_score_loss(y, p))


def cv_oof(name, X, y, repeats=5):
    oof = np.zeros((repeats, len(y)))
    for r in range(repeats):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED + r).split(X, y):
            m = make_pipeline(name, SEED).fit(X.iloc[tr], y.iloc[tr])
            oof[r, te] = m.predict_proba(X.iloc[te])[:, 1]
    return oof


def main(a):
    raw = pd.read_excel(a.data)
    if a.rekap:  # label ulang dari Rekap_All (No HP + nama), menggantikan kolom "Lanjut Interview" hasil XLOOKUP nama
        raw["Lanjut Interview"] = relabel(raw, a.rekap).astype(bool)
    geo = GeoResolver()
    feat = dedupe_applicants(build_features(raw, geo, DEFAULT_PLACEMENT))
    rel = feat[in_windows(feat["_ts"], RELIABLE_WINDOWS)].reset_index(drop=True)
    X, y = prepare_X(rel), rel["_y"].astype(int)
    print(f"Pelamar unik total={len(feat)} | dipakai (label andal)={len(rel)} | positif={y.sum()} ({y.mean():.1%})")

    results, oofs = {}, {}
    for name in ("logreg", "rf", "xgb"):
        oof = cv_oof(name, X, y)
        p = oof.mean(0)
        thr = best_threshold(y, p)
        per_rep = [roc_auc_score(y, o) for o in oof]
        results[name] = dict(**metrics(y, p, thr), threshold=thr, auc_sd=float(np.std(per_rep)))
        oofs[name] = p
        print(f"{name:7s} AUC={results[name]['auc']:.3f}±{results[name]['auc_sd']:.3f} AP={results[name]['ap']:.3f} "
              f"F1={results[name]['f1']:.3f} P={results[name]['precision']:.2f} R={results[name]['recall']:.2f} thr={thr:.2f}")
    # baseline: hanya jarak
    from sklearn.linear_model import LogisticRegression
    d = rel[["jarak_km"]].clip(upper=100).fillna(rel["jarak_km"].median())
    pj = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(d, y):
        pj[te] = LogisticRegression().fit(d.iloc[tr], y.iloc[tr]).predict_proba(d.iloc[te])[:, 1]
    results["baseline_jarak_saja"] = dict(auc=roc_auc_score(y, pj), ap=average_precision_score(y, pj))
    print(f"baseline jarak-saja AUC={results['baseline_jarak_saja']['auc']:.3f}  (prevalensi={y.mean():.3f})")

    # uji lintas waktu: model dilatih di RELIABLE_WINDOWS, diuji di CHECK_WINDOW (periode yang tidak dipakai training)
    chk = feat[in_windows(feat["_ts"], [CHECK_WINDOW])].reset_index(drop=True)
    temporal = {}
    if len(chk) >= 20 and chk["_y"].nunique() > 1:
        Xc, yc = prepare_X(chk), chk["_y"].astype(int)
        for name in ("logreg", "rf", "xgb"):
            p = make_pipeline(name, SEED).fit(X, y).predict_proba(Xc)[:, 1]
            temporal[name] = dict(n_test=len(chk), pos_rate=float(yc.mean()), auc=roc_auc_score(yc, p), ap=average_precision_score(yc, p))
            print(f"[periode pembanding {CHECK_WINDOW[0][:7]}..{CHECK_WINDOW[1][:7]}] {name:7s} n={len(chk)} AUC={temporal[name]['auc']:.3f}")
    # pilih model terbaik berdasarkan AP CV (lalu AUC)
    best = max(("logreg", "rf", "xgb"), key=lambda k: (results[k]["ap"] + results[k]["auc"]))
    if a.model:
        best = a.model
    thr = results[best]["threshold"]
    print("Model dipilih:", best, "threshold:", round(thr, 2))

    final = make_pipeline(best, SEED).fit(X, y)
    expl = Explainer(final, background_X=X)
    joblib.dump(dict(pipeline=final, explainer_state=expl.to_state(), versions=lib_versions(), threshold=thr, rates=compute_rates(rel, y), model_name=best, features=FEATURES,
                     train_info=dict(n=len(rel), pos=int(y.sum()), windows=RELIABLE_WINDOWS, placement=DEFAULT_PLACEMENT)), MODEL_PATH)

    # importance global (rata-rata |SHAP| per tema)
    contrib = expl.contributions(X)
    imp = contrib.abs().mean().sort_values(ascending=False).rename("mean_abs_shap")
    imp.to_csv(ROOT / "outputs" / "feature_importance.csv")
    print("\nPengaruh per tema (mean |SHAP|):\n", imp.round(3).to_string())
    out = dict(n_train=len(rel), n_pos=int(y.sum()), cv=results, temporal=temporal, chosen=best, threshold=thr)
    (ROOT / "outputs" / "metrics.json").write_text(json.dumps(out, indent=2, default=float))
    oof = pd.DataFrame(dict(ts=rel["_ts"], y=y, **{k: v for k, v in oofs.items()}))
    oof.to_csv(ROOT / "outputs" / "oof_predictions.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--rekap", help="path Rekap_All.xlsx untuk relabel lewat No HP")
    ap.add_argument("--model", choices=["logreg", "rf", "xgb"])
    main(ap.parse_args())
