"""Tes cepat: python -m pytest tests -q   (atau python tests/test_pipeline.py)"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import pandas as pd

from geo import GeoResolver, haversine_km
from sheet_sync import build_output_rows, make_keys, pending_rows


def test_haversine():
    assert abs(haversine_km(-6.2, 106.8, -6.2, 106.8)) < 1e-9
    assert 400 < haversine_km(-6.2, 106.8, -7.8, 110.4) < 500  # Jakarta-Yogya ~ 430 km


def test_geo_resolver():
    g = GeoResolver()
    r = g.resolve("Banten", "Tangerang Selatan", "Pondok aren", "", "")
    assert r["level"] == "kecamatan" and "Pondok Aren" in r["match"]
    r = g.resolve("", "Tanggerang", "Cisauk", "", "")
    assert "Cisauk" in r["match"] and g.distance_km(r["lat"], r["lng"], "Cisauk") < 0.01
    r = g.resolve("", "", "", "", "ngawur total xyz")
    assert r["level"] == "tidak_terbaca"


def test_sync_logic_and_predict():
    from predict import Predictor
    form = pd.DataFrame({
        "Timestamp": ["24/06/2024 16:59:59", "25/06/2024 10:00:00"], "Nama Lengkap": ["A", "B"],
        "Nomor KTP": ["3671115506010005", "3671115506010006"],
        "Tanggal Lahir": ["15/06/2001", "01/01/1995"], "Status Pernikahan": ["Belum Menikah", "Menikah Sudah Memiliki Anak"],
        "Provinsi Tinggal Saat Ini": ["Banten", "Jawa Barat"], "Kabupaten / Kotamadya Tinggal Saat ini": ["Kabupaten Tangerang", "Kota Bandung"],
        "Kecamatan Tinggal Saat Ini": ["Cikupa", "Ujungberung"], "Kelurahan Tinggal Saat Ini": ["Cibadak", "Pasanggrahan"],
        "Pendidikan Terakhir": ["SMA/SMK", "S1"], "Nilai Terakhir": ["80", "3.5"], "Lokasi Penempatan": ["Cisauk", "Cisauk"],
    })
    keys = make_keys(form)
    todo = pending_rows(form, {keys.iloc[0]})
    assert len(todo) == 1 and todo.iloc[0]["Nama Lengkap"] == "B"
    rows = build_output_rows(Predictor(), pending_rows(form, set()))
    assert len(rows) == 2 and rows[0][3] in ("LANJUT INTERVIEW", "TIDAK LANJUT")
    assert rows[1][6] > 100  # Bandung jauh dari Cisauk (km)


if __name__ == "__main__":
    for f in (test_haversine, test_geo_resolver, test_sync_logic_and_predict):
        f(); print("OK", f.__name__)
