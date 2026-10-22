"""Prediksi untuk pelamar baru.

Dipakai sebagai modul:  from predict import Predictor; Predictor().predict(df_form_responses)
Atau CLI uji        :  python predict.py --data pelamar.xlsx --out hasil.xlsx
"""
import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from config import DEFAULT_PLACEMENT, MODEL_PATH  # noqa: E402
from features import build_features  # noqa: E402
from geo import GeoResolver  # noqa: E402
from modeling import Explainer, explain_rows, prepare_X  # noqa: E402

OUT_COLS = ["AI_Rekomendasi", "AI_Skor", "AI_Keyakinan", "AI_Jarak_km", "AI_Domisili_Terbaca", "AI_Alasan_Utama", "AI_Faktor_Lain"]


class Predictor:
    def __init__(self, model_path=MODEL_PATH):
        art = joblib.load(model_path)
        self.pipe, self.thr = art["pipeline"], art["threshold"]
        self.expl = Explainer(self.pipe, state=art["explainer_state"])
        self.versions = art.get("versions", {})
        self.info = art["train_info"]
        self.rates = art.get("rates")
        self.geo = GeoResolver()

    def predict(self, raw: pd.DataFrame, placement: str = DEFAULT_PLACEMENT) -> pd.DataFrame:
        raw = raw.reset_index(drop=True)
        feat = build_features(raw, self.geo, placement)
        X = prepare_X(feat)
        proba = self.pipe.predict_proba(X)[:, 1]
        contrib = self.expl.contributions(X)
        place = raw.filter(regex="(?i)lokasi penempatan").iloc[:, 0].astype(str).str.strip().tolist() if raw.filter(regex="(?i)lokasi penempatan").shape[1] else [placement] * len(raw)
        rows = []
        for i in range(len(raw)):
            r = explain_rows(contrib.iloc[[i]], feat.iloc[[i]], proba[[i]], self.thr, place[i] or placement, rates=self.rates)[0]
            rows.append(r)
        res = pd.DataFrame(rows)
        out = pd.DataFrame({
            "AI_Rekomendasi": res["rekomendasi"], "AI_Skor": res["skor"], "AI_Keyakinan": res["keyakinan"],
            "AI_Jarak_km": feat["jarak_km"].round(1), "AI_Domisili_Terbaca": feat["geo_match"],
            "AI_Alasan_Utama": res["alasan_utama"], "AI_Faktor_Lain": res["faktor_pertimbangan"],
        })
        return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="hasil_prediksi.xlsx")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    df = pd.read_excel(a.data) if a.data.endswith(("xlsx", "xls")) else pd.read_csv(a.data)
    if a.limit:
        df = df.head(a.limit)
    res = Predictor().predict(df)
    pd.concat([df.reset_index(drop=True), res], axis=1).to_excel(a.out, index=False)
    print(res["AI_Rekomendasi"].value_counts())
    print("Tersimpan:", a.out)
