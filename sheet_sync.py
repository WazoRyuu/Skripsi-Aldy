"""Sinkronisasi Google Sheet (jawaban Google Form) <-> model.

Alur: baca tab jawaban form -> cari pelamar yang belum diproses -> prediksi -> tulis ke tab terpisah "Prediksi AI".
Tab hasil sengaja DIPISAH dari tab jawaban supaya Google Form tidak bentrok saat menambah kolom/baris.

Contoh:
  python sheet_sync.py --sheet-id <ID_SPREADSHEET> --creds service_account.json            # sekali jalan
  python sheet_sync.py --sheet-id <ID_SPREADSHEET> --creds service_account.json --loop 300 # cek tiap 5 menit
"""
import argparse
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import OUT_COLS, Predictor  # noqa: E402

OUT_TAB = "Prediksi AI"
HEADER = ["Key", "Timestamp", "Nama Lengkap"] + OUT_COLS + ["Diproses Pada"]


def _col(df: pd.DataFrame, prefix: str) -> pd.Series:
    for c in df.columns:
        if re.sub(r"\s+", " ", str(c)).strip().lower().startswith(prefix):
            return df[c].astype(str)
    return pd.Series([""] * len(df), index=df.index)


def make_keys(df: pd.DataFrame) -> pd.Series:
    ts = _col(df, "timestamp").str.strip()
    ktp = _col(df, "nomor ktp").map(lambda x: re.sub(r"\D", "", x))
    nik = _col(df, "nik").map(lambda x: re.sub(r"\D", "", x))  # header form baru
    ktp = ktp.where(ktp != "", nik)
    nama = _col(df, "nama lengkap").str.strip().str.lower()
    return ts + "|" + ktp.where(ktp != "", nama)


def pending_rows(form: pd.DataFrame, done_keys: set) -> pd.DataFrame:
    form = form.copy()
    form["_key"] = make_keys(form)
    ok = (_col(form, "timestamp").str.strip() != "") & (_col(form, "nama lengkap").str.strip() != "")
    return form[ok & ~form["_key"].isin(done_keys)]


def build_output_rows(P: Predictor, todo: pd.DataFrame) -> list[list]:
    res = P.predict(todo.drop(columns="_key"))
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for k, ts, nm, (_, r) in zip(todo["_key"], _col(todo, "timestamp"), _col(todo, "nama lengkap"), res.iterrows()):
        rows.append([k, ts, nm] + [("" if pd.isna(r[c]) else r[c]) for c in OUT_COLS] + [now])
    return rows


def run_once(sheet_id: str, creds: str, form_tab: str | None, P: Predictor) -> int:
    import gspread

    gc = gspread.service_account(filename=creds)
    sh = gc.open_by_key(sheet_id)
    ws_form = sh.worksheet(form_tab) if form_tab else sh.get_worksheet(0)
    try:
        ws_out = sh.worksheet(OUT_TAB)
    except gspread.WorksheetNotFound:
        ws_out = sh.add_worksheet(OUT_TAB, rows=2000, cols=len(HEADER))
        ws_out.append_row(HEADER)
        ws_out.freeze(rows=1)

    values = ws_form.get_all_values()
    if len(values) < 2:
        return 0
    form = pd.DataFrame(values[1:], columns=values[0])
    done = set(ws_out.col_values(1)[1:])
    todo = pending_rows(form, done)
    if todo.empty:
        return 0
    for i in range(0, len(todo), 50):  # batch kecil supaya aman dari limit API
        ws_out.append_rows(build_output_rows(P, todo.iloc[i:i + 50]), value_input_option="RAW")
    return len(todo)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet-id", required=True)
    ap.add_argument("--creds", required=True, help="path JSON service account")
    ap.add_argument("--form-tab", help="nama tab jawaban form (default: tab pertama)")
    ap.add_argument("--loop", type=int, default=0, help="detik antar pengecekan; 0 = sekali jalan")
    a = ap.parse_args()
    P = Predictor()
    while True:
        try:
            n = run_once(a.sheet_id, a.creds, a.form_tab, P)
            print(f"[{datetime.now():%H:%M:%S}] pelamar baru diproses: {n}", flush=True)
        except Exception as e:  # jangan mati karena error jaringan/API sesaat
            print("ERROR:", repr(e), flush=True)
            if not a.loop:
                raise
        if not a.loop:
            break
        time.sleep(a.loop)
