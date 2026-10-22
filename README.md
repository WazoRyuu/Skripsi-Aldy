# Prediksi Kelayakan Lanjut Interview — Tele Marketing

**Dua output:** (1) sinkronisasi ke Google Sheet (`sheet_sync.py` / notebook), terhubung ke data perusahaan; (2) **website Streamlit** (`app.py`) dengan input manual, *standalone* (tidak terhubung ke database perusahaan, tidak menyimpan data).

Membaca jawaban Google Form pelamar → menghitung jarak domisili ke lokasi penempatan → memberi
**rekomendasi (LANJUT / TIDAK LANJUT)**, skor, dan **alasan** per pelamar → menulis hasilnya ke Google Sheet.

## Struktur
```
app.py            website Streamlit: input manual -> peluang lolos interview + jarak km (standalone)
train.py          latih + evaluasi 3 model (LogReg, RandomForest, XGBoost), simpan model terbaik
predict.py        class Predictor (dipakai sheet_sync) + CLI uji ke file Excel
sheet_sync.py     tarik pelamar baru dari Google Sheet -> prediksi -> tulis ke tab "Prediksi AI"
src/config.py     periode data training, lokasi penempatan default
src/geo.py        domisili (teks bebas) -> kecamatan -> koordinat -> jarak km (haversine)
src/labeling.py   label dari Rekap_All (No HP + nama)
src/features.py   cleaning + feature engineering (dipakai sama persis saat training & prediksi)
src/modeling.py   pipeline model + SHAP + pembuat kalimat alasan
data/             koordinat 7.265 kecamatan Indonesia (centroid polygon, cahyadsn/wilayah_boundaries, MIT)
outputs/          metrics.json, feature_importance.csv, oof_predictions.csv
tests/            tes logika geo, sync, dan prediksi
```

## Jalankan
```bash
pip install -r requirements-train.txt     # training / sync (app web cukup requirements.txt)
python train.py --data "Form Loker Tele Marketing (Jawaban) (1).xlsx" --rekap Rekap_All.xlsx   # latih ulang + relabel
python predict.py --data pelamar_baru.xlsx --out hasil.xlsx                # uji manual
python tests/test_pipeline.py                                              # tes
```

## Website Streamlit (standalone)
Pelamar/recruiter mengisi form manual (pendidikan, riwayat kerja, domisili lewat dropdown provinsi → kab/kota → kecamatan, dll).
Hasil: pesan peluang (tinggi / cukup baik / belum pasti / "Mohon maaf, ... belum sesuai") + jarak ke lokasi penempatan.
Mode **Recruiter** (sidebar) menambah skor dan alasan rinci. Aplikasi **tidak terhubung ke database/Google Sheet**, tidak menyimpan
input, dan tidak meminta nama/KTP/HP. Fitur baca CV belum ada (prioritas rendah).

Jalankan lokal: `pip install -r requirements.txt` lalu `streamlit run app.py`.

### Deploy (Streamlit Community Cloud, gratis)
1. Buat repo GitHub berisi: `app.py`, `src/`, `data/`, `models/model_interview.joblib`, `requirements.txt`, `.streamlit/`.
   **Jangan** upload file `.xlsx` (data pelamar), `outputs/`, atau `service_account.json` (sudah ada di `.gitignore`).
2. Buka share.streamlit.io → New app → pilih repo → Main file `app.py` → Advanced settings → Python 3.12 → Deploy.
3. Kalau model dilatih ulang (mis. di Colab), ganti `models/model_interview.joblib` dan samakan versi di `requirements.txt` dengan
   isi `outputs/model_versions.txt` (model scikit-learn bisa gagal dimuat bila versinya beda).
4. Pengaturan akses (publik / hanya email tertentu) ada di menu Share pada Streamlit Cloud; cek sebelum membagikan link.

## Sambung ke Google Sheet (sekali setup)
1. Google Cloud Console → buat project → aktifkan **Google Sheets API** & **Google Drive API**.
2. IAM → Service Accounts → buat akun → Keys → **Add key (JSON)** → simpan `service_account.json`.
3. Buka Google Sheet jawaban form → **Share** ke email service account (`...iam.gserviceaccount.com`) sebagai *Editor*.
4. Jalankan (ID sheet = bagian di URL antara `/d/` dan `/edit`):
```bash
python sheet_sync.py --sheet-id <ID> --creds service_account.json --loop 300   # cek tiap 5 menit
```
Hasil ditulis ke tab baru **Prediksi AI** (terpisah dari tab jawaban form supaya Form tidak bentrok):
`Timestamp, Nama, AI_Rekomendasi, AI_Skor, AI_Keyakinan, AI_Jarak_km, AI_Domisili_Terbaca, AI_Alasan_Utama, AI_Faktor_Lain`.
Pelamar yang sudah diproses tidak diproses ulang (kunci = Timestamp + KTP). Agar jalan terus tanpa laptop menyala:
jadwalkan di server kecil / cron / GitHub Actions (cron, tanpa `--loop`).

## Jalan terus tanpa laptop/Colab (GitHub Actions)
`.github/workflows/sync-sheet.yml` menjalankan `sheet_sync.py` otomatis di server GitHub (default: tiap 15 menit, 07.00-19.00 WIB, Senin-Sabtu).
1. Taruh project di repo GitHub **private** (sama dengan repo untuk Streamlit).
2. Repo -> Settings -> Secrets and variables -> Actions -> New repository secret:
   `SHEET_ID` (id sheet), `GCP_SA_JSON` (isi lengkap `service_account.json`), opsional `FORM_TAB`.
3. Tab Actions -> sync-sheet -> **Run workflow** untuk tes manual. Lihat log: `pelamar baru diproses: N`.
Catatan: jadwal GitHub bisa telat beberapa menit; repo private gratis punya kuota ±2000 menit/bulan (jadwal ini ±1250).
Mau real-time (langsung saat form disubmit) butuh server/API yang hidup terus (mis. Cloud Run) — di luar paket ini.

## Cara kerja jarak
Teks Provinsi/Kab-Kota/Kecamatan/Kelurahan/Alamat dinormalisasi (typo & singkatan: "Tanggerang", "tangsel", dst), dicocokkan
ke daftar kecamatan resmi (fuzzy match), lalu jarak **garis lurus** dihitung dari centroid kecamatan ke titik penempatan.
Jarak jalan biasanya ±1,3–1,6× lebih panjang. Titik penempatan default = centroid Kecamatan Cisauk; untuk titik kantor yang
presisi isi `PLACEMENTS` di `src/geo.py`. Lokasi penempatan lain: cukup isi di kolom "Lokasi Penempatan" form
(nama kecamatan), atau tambahkan koordinatnya di `PLACEMENTS`.
Cakupan pada data ini: 98,9% terbaca sampai level kecamatan, sisanya level kab/kota.

## Hasil evaluasi (data 2024-06..2024-11, 402 pelamar unik, 43,8% tembus)
Label dibuat ulang dari Rekap_All lewat **No HP (+ kemiripan nama) atau nama persis** (`src/labeling.py`), sehingga pelamar yang
ejaan namanya beda tetap terdeteksi (+8 positif dibanding XLOOKUP nama saja).

| Model | AUC | PR-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| **Logistic Regression (dipakai)** | 0,721 | 0,661 | 0,626 | 0,64 | 0,61 |
| Random Forest | 0,721 | 0,657 | 0,617 | 0,68 | 0,56 |
| XGBoost | 0,709 | 0,662 | 0,602 | 0,66 | 0,55 |
| Baseline jarak saja | 0,608 | 0,512 | – | – | – |

5-fold CV x 5 ulangan; threshold (0,54) dipilih dengan Youden's J. Logistic Regression dipilih karena performanya setara
tetapi paling mudah dijelaskan. Turunkan threshold bila lebih takut kehilangan kandidat bagus (recall naik, precision turun).

## Keterbatasan penting (baca sebelum dipakai untuk keputusan)
1. **Tidak ada rekap 2025** (sudah dikonfirmasi: Rekap_All memang tidak memuat 2025; relabel lewat HP pun tidak memulihkannya).
   Pelamar 2024-12..2025-11 berlabel False karena *tidak tercatat*, bukan *ditolak*, jadi tidak dipakai training
   (`RELIABLE_WINDOWS` di `src/config.py`).
2. **Pola 2024 tidak berlaku di 2025-12..2026-01** (AUC 0,43 pada 79 pelamar). Di 2024 yang tembus tinggal jauh lebih dekat
   (rata-rata 15,5 km vs 38,9 km yang tidak tembus); di Des 2025-Jan 2026 terbalik (19,2 km vs 11,5 km) dan 65-80% pelamar masuk rekap.
   Dugaan: alur 2026 berubah (mis. form diisi setelah kandidat dihubungi), sehingga "ada di rekap" tidak lagi berarti "lolos
   seleksi berkas". Perlu dikonfirmasi ke PT XYZ. Model ini mereplikasi kebiasaan seleksi 2024.
3. File form berakhir 2026-01-29, sedangkan Rekap_All berisi interview sampai 2026-09. Export form terbaru akan menambah
   data berlabel periode terkini.
4. AUC ±0,72 = model *membantu mengurutkan prioritas*, bukan pengganti penilaian recruiter. Pelamar berkeyakinan "Borderline"
   sebaiknya selalu ditinjau manual.
5. Jenis kelamin, kehamilan, tempat lahir, nama/KTP/HP **sengaja tidak dipakai**. Usia & status pernikahan dipakai karena
   merupakan persyaratan sebagian posisi PT XYZ (keputusan peneliti); pastikan sesuai kebijakan perusahaan.
6. Alasan dari model bersifat **korelasi historis**, bukan sebab-akibat.
