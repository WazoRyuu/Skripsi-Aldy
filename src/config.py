"""Konfigurasi proyek."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model_interview.joblib"

# Periode (berdasarkan Timestamp form) yang labelnya DAPAT DIPERCAYA untuk training.
# Alasan: pelamar 2024-12..2025-11 hampir semuanya berlabel False (2-6% True vs 40%+ di 2024) karena rekap 2025 belum
# ada -> itu "label hilang", bukan "ditolak". Jangan dipakai training.
# Setelah data rekap 2025 diterima/di-relabel: tambahkan periodenya di sini lalu jalankan ulang train.py.
RELIABLE_WINDOWS = [("2024-06-01", "2024-11-30")]
# Periode pembanding (TIDAK dipakai training) untuk uji "apakah model masih berlaku di periode baru".
CHECK_WINDOW = ("2025-12-01", "2026-01-31")

DEFAULT_PLACEMENT = "Cisauk"
SEED = 42
