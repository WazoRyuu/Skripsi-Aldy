"""Bangun SATU Excel induk berisi SEMUA pengisian form pelamar Tele Marketing (2024 - 2026).

  python build_master.py --v1 Form_Loker_Tele_Marketing_Jawaban.xlsx --v2 Kuis_tanpa_judul.xlsx \\
                         --v3 Form_Data_Diri_Kandidat.xlsx --rekap Rekap_All.xlsx --screening Screening.xlsx --out Data_Induk_Pelamar.xlsx

Aturan (sengaja KOSONG bila tidak diketahui, agar perusahaan bisa melengkapinya):
  Status Lanjut Interview
    "Lanjut Interview" : pelamar ada di Rekap_All dan tanggal interview jatuh 1 hari sebelum s.d. 90 hari sesudah pengisian,
                         atau ada di rekap tanpa tanggal.
    "Tidak Lanjut"     : pelamar TIDAK ada di Rekap_All, dan pengisian berada di periode yang cakupan rekapnya lengkap.
    kosong             : periode rekap tidak lengkap (Des 2024 - Nov 2025, Feb-Mar 2026), pengisian terlalu baru, pelamar melamar
                         ulang setelah pernah interview, atau Form Data Diri tanpa catatan di rekap.
  Poin Total / Hasil Poin: hanya untuk Kuis 2026, diambil dari Screening.xlsx (dicocokkan lewat NIK / No HP + nama dan waktu).
"""
import argparse
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
warnings.filterwarnings("ignore")

from harmonize import Geo, Matcher, adapt_v1, adapt_v2, adapt_v3, load_rekap_dated, person_key  # noqa: E402

# periode pengisian yang cakupan Rekap_All-nya lengkap -> "tidak ada di rekap" boleh dibaca sebagai "Tidak Lanjut"
COVERED = {"v1": [("2024-06-01", "2024-11-30"), ("2025-12-01", "2026-01-31")], "v2": [("2026-04-01", "2026-08-31")], "v3": []}
SRC_NAME = {"v1": "Form Lama (Google Form)", "v2": "Kuis 2026 (MS Forms)", "v3": "Form Data Diri 2026 (MS Forms)", "scr": "Kuis 2026 (hanya ada di Screening)"}
LANJUT, TIDAK = "Lanjut Interview", "Tidak Lanjut"


def covered(era, ts, cutoff):
    return any(pd.Timestamp(a) <= ts < pd.Timestamp(b) + pd.Timedelta(days=1) and ts <= cutoff for a, b in COVERED.get(era, []))


def status_row(era, ts, dates, cutoff, lag=90):
    """-> (status, keterangan, tanggal_interview)."""
    if dates is None:
        if covered(era, ts, cutoff):
            return TIDAK, "tidak ada di Rekap_All (periode rekap lengkap)", pd.NaT
        if era == "v3":
            return None, "Form Data Diri diisi setelah kandidat dipanggil; tidak ada di rekap", pd.NaT
        if ts > cutoff:
            return None, f"pengisian terlalu baru (Rekap_All berakhir {cutoff + pd.Timedelta(days=14):%d/%m/%Y})", pd.NaT
        return None, "periode ini tidak tercakup Rekap_All", pd.NaT
    known = [d for d in dates if pd.notna(d)]
    if not known:
        return LANJUT, "ada di Rekap_All (tanggal interview kosong)", pd.NaT
    ok = [d for d in known if ts.normalize() - pd.Timedelta(days=1) <= d <= ts.normalize() + pd.Timedelta(days=lag)]
    if ok:
        return LANJUT, "ada di Rekap_All", min(ok)
    return None, "melamar ulang setelah pernah interview (tidak jelas)", min(known)


def load_screening(path):
    s = pd.read_excel(path, sheet_name="Sheet1")
    s.columns = [c.strip() for c in s.columns]
    s["ts"] = pd.to_datetime(s["Timestamp"], format="%d-%m-%Y %H:%M:%S", errors="coerce")
    s = s[s["ts"].notna()].reset_index(drop=True)  # baris uji (utcNow()) tidak punya waktu -> dibuang
    s["Score"] = pd.to_numeric(s["Score"], errors="coerce")
    s["key"] = [person_key(a, b, c) for a, b, c in zip(s["NIK"], s["Nama"], s["WA"])]
    return s


def attach_points(u, scr):
    """Cocokkan baris kuis (v2) dengan Screening: kunci orang sama dan selisih waktu <= 1 hari, satu-satu terdekat."""
    u["poin"], u["hasil_poin"] = np.nan, None
    k = u[u["era"] == "v2"][["key", "ts_selesai"]].reset_index().rename(columns={"index": "iu"})
    m = k.merge(scr[["key", "ts", "Score", "Hasil"]].reset_index().rename(columns={"index": "is_"}), on="key")
    m["dt"] = (m["ts_selesai"] - m["ts"]).abs()
    m = m[m["dt"] <= pd.Timedelta(days=1)].sort_values("dt")
    used_u, used_s = set(), set()
    for r in m.itertuples():
        if r.iu in used_u or r.is_ in used_s:
            continue
        used_u.add(r.iu), used_s.add(r.is_)
        u.at[r.iu, "poin"], u.at[r.iu, "hasil_poin"] = r.Score, str(r.Hasil).strip().title()
    return u, sorted(set(range(len(scr))) - used_s)


COLS = [
    ("id", "ID", "Nomor urut baris"),
    ("sumber", "Sumber Form", "Form asal pengisian"),
    ("ts", "Waktu Pengisian", "Waktu pelamar mengisi form"),
    ("nama", "Nama", "IDENTITAS - bukan fitur model"), ("hp", "No HP", "IDENTITAS - kunci pencocokan ke rekap"),
    ("ktp", "NIK/KTP", "IDENTITAS - kunci data unik"),
    ("ke", "Pengisian Ke-", "urutan pengisian oleh orang yang sama (berdasarkan NIK, atau No HP + nama)"),
    ("dari", "Jumlah Pengisian", "berapa kali orang ini mengisi form (semua form)"),
    ("status", "Status Lanjut Interview", "ISI: 'Lanjut Interview' atau 'Tidak Lanjut'. Kosong = belum diketahui, mohon dilengkapi perusahaan"),
    ("ket_status", "Keterangan Status", "alasan status terisi / kosong"),
    ("tgl_int", "Tanggal Interview (Rekap)", "Tanggal interview di Rekap_All"),
    ("poin", "Poin Total", "Skor kuis 0-100 (Screening.xlsx). Kosong = tidak ada skor, mohon dilengkapi perusahaan bila ada"),
    ("hasil_poin", "Hasil Poin", "Lolos (75-100) / Pertimbangan (50-74) / Tidak Lolos (0-49)"),
    ("ket_poin", "Keterangan Poin", "alasan poin terisi / kosong"),
    ("usia", "Usia", "tahun, dari tanggal lahir (cadangan: titik tengah pilihan usia di kuis)"),
    ("pend_level", "Pendidikan", "SMA/SMK, D1, D3, D4, S1, S2, SMP, Lainnya"),
    ("fresh_grad", "Fresh Graduate", "1 = belum pernah bekerja"),
    ("exp_tele", "Pengalaman Tele (ada)", "Telemarketing/Telecollection/Deskcollection/Customer Care; 1 = pernah"),
    ("exp_tele_ord", "Pengalaman Tele (lama)", "0 tidak ada, 1 < 1 tahun, 2 >= 1 tahun, kosong = lama tidak diketahui"),
    ("exp_finance", "Pengalaman Finance (ada)", "Multifinance/Perbankan; 1 = pernah"),
    ("exp_finance_ord", "Pengalaman Finance (lama)", "skala sama dengan Pengalaman Tele (lama)"),
    ("exp_sales", "Pengalaman Sales", "1 = pernah (tidak ditanyakan di kuis -> kosong)"),
    ("jarak_km", "Jarak ke Cisauk (km)", "garis lurus; lihat Sumber Jarak"),
    ("jarak_src", "Sumber Jarak", "kecamatan (presisi) / kab/kota (kasar) / bracket (titik tengah pilihan jarak di kuis) / tidak_terbaca"),
    ("geo_match", "Domisili Terbaca", "hasil pembacaan domisili"),
    ("ext_jarak_bracket_km", "[Kuis] Jarak Pilihan Pelamar (km)", "hanya kuis"),
    ("ext_ms_office", "[Kuis] MS Office", "0 tidak, 1 jarang, 2 sangat terbiasa"),
    ("ext_siap_join", "[Kuis] Siap Join", "1 = siap segera"),
    ("ext_menikah", "[Lama] Menikah", "hanya form lama"), ("ext_punya_anak", "[Lama] Punya Anak", "hanya form lama"),
    ("ext_nilai_norm", "[Lama] Nilai (0-100)", "hanya form lama"), ("ext_kuliah_sekarang", "[Lama] Sedang Kuliah", "hanya form lama"),
]


def build(a):
    geo = Geo()
    v1, v2, v3 = (pd.read_excel(p) for p in (a.v1, a.v2, a.v3))
    parts = [adapt_v1(v1, geo), adapt_v2(v2, geo), adapt_v3(v3, geo)]
    # waktu selesai kuis dipakai hanya untuk mencocokkan skor
    c2 = {re.sub(r"\s+", " ", str(k)).strip().lower(): k for k in v2.columns}
    parts[1]["ts_selesai"] = pd.to_datetime(v2[c2["waktu selesai"]], errors="coerce")
    u = pd.concat(parts, ignore_index=True)
    u = u[u["ts"].notna() & u["nama"].notna()].reset_index(drop=True)
    u["key"] = [person_key(k, n, h) for k, n, h in zip(u["ktp"], u["nama"], u["hp"])]

    rek = load_rekap_dated(a.rekap)
    M = Matcher(rek)
    cutoff = rek["d"].max() - pd.Timedelta(days=14)
    st = [status_row(e, t, M.dates(n, h), cutoff) for e, t, n, h in zip(u["era"], u["ts"], u["nama"], u["hp"])]
    u["status"], u["ket_status"], u["tgl_int"] = [s[0] for s in st], [s[1] for s in st], [s[2] for s in st]

    scr = load_screening(a.screening)
    u, sisa = attach_points(u, scr)
    u["ket_poin"] = np.where(u["poin"].notna(), "dari Screening.xlsx",
                             np.where(u["era"] == "v2", "kuis ini tidak ditemukan di Screening.xlsx", "form ini tidak memiliki sistem poin"))

    # baris yang HANYA ada di Screening (tidak ada di ekspor kuis): tambahkan, isian terbatas
    extra = []
    for i in sisa:
        r = scr.loc[i]
        ds = M.dates(r["Nama"], r["WA"])
        s = status_row("v2", r["ts"], ds, cutoff)
        extra.append(dict(era="scr", ts=r["ts"], nama=r["Nama"], hp=r["WA"], ktp=r["NIK"], key=r["key"], status=s[0], ket_status=s[1], tgl_int=s[2],
                          poin=r["Score"], hasil_poin=str(r["Hasil"]).strip().title(), ket_poin="dari Screening.xlsx (tidak ada di ekspor kuis)",
                          pend_level=None, exp_tele_ord=None))
    if extra:
        u = pd.concat([u, pd.DataFrame(extra)], ignore_index=True)

    u["sumber"] = u["era"].map(SRC_NAME)
    u = u.sort_values(["ts"]).reset_index(drop=True)
    u["ke"] = u.groupby("key").cumcount() + 1
    u["dari"] = u.groupby("key")["key"].transform("size")
    u["id"] = range(1, len(u) + 1)
    for c, _n, _e in COLS:
        if c not in u.columns:
            u[c] = None
    out = u[[c for c, _n, _e in COLS]].copy()
    out.columns = [n for _c, n, _e in COLS]
    out["Waktu Pengisian"] = pd.to_datetime(out["Waktu Pengisian"]).dt.tz_localize(None)
    out["Tanggal Interview (Rekap)"] = pd.to_datetime(out["Tanggal Interview (Rekap)"], errors="coerce").dt.tz_localize(None)
    for c in out.columns:
        if out[c].dtype == float:
            out[c] = out[c].round(2)
    return out


def to_excel(out, path):
    from openpyxl import load_workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation

    with pd.ExcelWriter(path, engine="openpyxl", datetime_format="DD/MM/YYYY HH:MM") as xw:
        out.to_excel(xw, sheet_name="Data_Pelamar", index=False)
        pd.DataFrame({"Kolom": [n for _c, n, _e in COLS], "Penjelasan": [e for _c, _n, e in COLS]}).to_excel(xw, sheet_name="Kamus_Kolom", index=False)
    wb = load_workbook(path)
    base, head = Font(name="Arial", size=10), PatternFill("solid", fgColor="1F3864")
    ws = wb["Data_Pelamar"]
    n = ws.max_row
    for row in ws.iter_rows():
        for c in row:
            c.font = base
    for c in ws[1]:
        c.font, c.fill, c.alignment = Font(name="Arial", size=10, bold=True, color="FFFFFF"), head, Alignment(wrap_text=True, vertical="center")
    ws.row_dimensions[1].height = 42
    ws.freeze_panes = "E2"
    ws.auto_filter.ref = ws.dimensions
    cols = list(out.columns)
    idx = {c: i + 1 for i, c in enumerate(cols)}
    fill_in = PatternFill("solid", fgColor="FFF2CC")  # kolom yang perlu dilengkapi perusahaan
    for c in ("Status Lanjut Interview", "Poin Total", "Hasil Poin"):
        ws.cell(1, idx[c]).fill = PatternFill("solid", fgColor="BF8F00")
        for r in range(2, n + 1):
            ws.cell(r, idx[c]).fill = fill_in
    for i, c in enumerate(cols, 1):
        w = {"Nama": 26, "Sumber Form": 30, "Keterangan Status": 44, "Keterangan Poin": 38, "Domisili Terbaca": 34, "NIK/KTP": 19,
             "Status Lanjut Interview": 18}.get(c, 14)
        ws.column_dimensions[get_column_letter(i)].width = w
        if c.startswith(("Waktu", "Tanggal")):
            for r in range(2, n + 1):
                ws.cell(r, i).number_format = "DD/MM/YYYY" if "Interview" in c else "DD/MM/YYYY HH:MM"
        if c in ("NIK/KTP", "No HP"):
            for r in range(2, n + 1):
                ws.cell(r, i).number_format = "@"
    dv = DataValidation(type="list", formula1='"Lanjut Interview,Tidak Lanjut"', allow_blank=True)
    dv.error, dv.errorTitle = "Pilih Lanjut Interview atau Tidak Lanjut", "Nilai tidak valid"
    ws.add_data_validation(dv)
    cl = get_column_letter(idx["Status Lanjut Interview"])
    dv.add(f"{cl}2:{cl}{n}")
    dv2 = DataValidation(type="list", formula1='"Lolos,Pertimbangan,Tidak Lolos"', allow_blank=True)
    ws.add_data_validation(dv2)
    cl2 = get_column_letter(idx["Hasil Poin"])
    dv2.add(f"{cl2}2:{cl2}{n}")
    dv3 = DataValidation(type="decimal", operator="between", formula1="0", formula2="100", allow_blank=True)
    ws.add_data_validation(dv3)
    cl3 = get_column_letter(idx["Poin Total"])
    dv3.add(f"{cl3}2:{cl3}{n}")

    kl = wb["Kamus_Kolom"]
    for row in kl.iter_rows():
        for c in row:
            c.font = base
    for c in kl[1]:
        c.font, c.fill = Font(name="Arial", size=10, bold=True, color="FFFFFF"), head
    kl.column_dimensions["A"].width, kl.column_dimensions["B"].width = 34, 110

    # ---- Ringkasan (rumus hidup) ----
    rs = wb.create_sheet("Ringkasan")
    rng = lambda name: f"Data_Pelamar!${get_column_letter(idx[name])}$2:${get_column_letter(idx[name])}${n}"
    rs["A1"], rs["A1"].font = "Ringkasan data induk pelamar Tele Marketing", Font(name="Arial", size=12, bold=True)
    hdr = ["Sumber Form", "Jumlah pengisian", "Lanjut Interview", "Tidak Lanjut", "Status kosong", "Poin terisi", "Poin kosong"]
    for j, h in enumerate(hdr, 1):
        c = rs.cell(3, j, h)
        c.font, c.fill, c.alignment = Font(name="Arial", size=10, bold=True, color="FFFFFF"), head, Alignment(wrap_text=True)
    names = list(SRC_NAME.values())
    for i, k in enumerate(names, 4):
        rs.cell(i, 1, k)
        rs.cell(i, 2, f"=COUNTIFS({rng('Sumber Form')},$A{i})")
        rs.cell(i, 3, f'=COUNTIFS({rng("Sumber Form")},$A{i},{rng("Status Lanjut Interview")},"Lanjut Interview")')
        rs.cell(i, 4, f'=COUNTIFS({rng("Sumber Form")},$A{i},{rng("Status Lanjut Interview")},"Tidak Lanjut")')
        rs.cell(i, 5, f"=B{i}-C{i}-D{i}")
        rs.cell(i, 6, f'=COUNTIFS({rng("Sumber Form")},$A{i},{rng("Poin Total")},">=0")')
        rs.cell(i, 7, f"=B{i}-F{i}")
    t = 4 + len(names)
    rs.cell(t, 1, "Total")
    for j in range(2, 8):
        col = get_column_letter(j)
        rs.cell(t, j, f"=SUM({col}4:{col}{t-1})")
    for r in range(4, t + 1):
        for j in range(1, 8):
            rs.cell(r, j).font = Font(name="Arial", size=10, bold=(r == t))
    notes = ["Catatan pengisian untuk perusahaan:",
             "- Sel berwarna kuning (Status Lanjut Interview, Poin Total, Hasil Poin) yang kosong = belum diketahui. Mohon dilengkapi bila datanya ada.",
             "- Status: pilih 'Lanjut Interview' bila pelamar lolos seleksi berkas / dipanggil interview, 'Tidak Lanjut' bila tidak. Dropdown tersedia.",
             "- Poin Total: skor kuis 0-100. Form lama (Google Form 2024 - Jan 2026) tidak memiliki sistem poin; isi hanya bila skor bisa dihitung.",
             "- Kolom Keterangan Status / Keterangan Poin menjelaskan kenapa sebuah sel kosong atau terisi.",
             "- Satu orang bisa mengisi beberapa kali: lihat 'Pengisian Ke-' dan 'Jumlah Pengisian'. Satu baris = satu pengisian form.",
             "- Rekap_All hanya memuat interview Jun 2024 - Nov 2024 dan Des 2025 - 02/09/2026; Des 2024 - Nov 2025 tidak tercakup (status dibiarkan kosong).",
             "- Kolom IDENTITAS (Nama, No HP, NIK) hanya untuk pencocokan; HAPUS sebelum dibagikan ke pihak lain / diunggah ke GitHub atau Colab."]
    for i, tx in enumerate(notes, t + 2):
        rs.cell(i, 1, tx).font = Font(name="Arial", size=10, bold=(i == t + 2))
    rs.column_dimensions["A"].width = 38
    for j in range(2, 8):
        rs.column_dimensions[get_column_letter(j)].width = 16
    rs.row_dimensions[3].height = 32
    wb.move_sheet("Ringkasan", offset=-2)
    wb.save(path)


def main():
    ap = argparse.ArgumentParser()
    for k in ("v1", "v2", "v3", "rekap", "screening"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--out", default="Data_Induk_Pelamar.xlsx")
    a = ap.parse_args()
    out = build(a)
    to_excel(out, a.out)
    s = out["Status Lanjut Interview"]
    print(f"{len(out)} baris | Lanjut {int((s == LANJUT).sum())} | Tidak Lanjut {int((s == TIDAK).sum())} | kosong {int(s.isna().sum())} | poin terisi {int(out['Poin Total'].notna().sum())}")


if __name__ == "__main__":
    main()
