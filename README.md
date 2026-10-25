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
Pelamar/recruiter mengisi form manual dalam 4 langkah (Data diri, Pendidikan, Pengalaman, Hasil; domisili lewat dropdown provinsi → kab/kota → kecamatan). Tampilan diatur lewat CSS di `app.py` dan tema di `.streamlit/config.toml` (tanpa emoji).
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
`Timestamp, Nama, AI_Rekomendasi, AI_Skor, AI_Keyakinan, AI_Jarak_km, AI_Domisili_Terbaca, AI_Alasan_Utama, AI_Faktor_Lain, AI_Catatan_Data`.
`AI_Catatan_Data` memberi peringatan bila input bermasalah (domisili/kecamatan tak terbaca atau dicocokkan otomatis, tanggal lahir/nilai tidak valid, pendidikan tak dikenali) supaya recruiter tahu kapan hasil perlu dicek manual.
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

## Pipeline data -> model
1. **Label**: pelamar dianggap "tembus" bila ada di Rekap_All (cocok No HP + nama mirip, atau nama persis) - `src/labeling.py`.
2. **Rename kolom form** ke nama pendek (cocok lewat awalan header, tahan perubahan kecil).
3. **Feature engineering** (`src/features.py`): usia (valid 16-65), menikah/punya anak, domisili -> kecamatan -> jarak km ke penempatan
   (`src/geo.py`), pendidikan dinormalkan, nilai/IPK disamakan ke skala 0-100, lama pengalaman -> ordinal, Ya/Tidak -> 1/0,
   flag kata kunci jabatan (sales/tele/collection), status kuliah.
4. **Dedupe**: 1 baris per pelamar (KTP; cadangan nama+HP), ambil submit terakhir, label = 1 bila pernah tembus.
5. **Filter periode** berlabel andal (`RELIABLE_WINDOWS`), 2024-06..2024-11 -> 402 pelamar.
6. **Persiapan model** (`prepare_X`): pilih 23 fitur, jarak dibatasi maks 100 km, kategori jadi teks.
7. **Preprocessing sklearn** (di dalam Pipeline, di-fit ulang per fold CV): angka -> median imputer (+StandardScaler utk
   Logistic Regression); biner -> isi 0; kategori -> One-Hot (kategori <8 pelamar digabung "infrequent") -> 54 kolom.
8. **Model**: Logistic Regression / Random Forest / XGBoost, dibandingkan dengan 5-fold CV x 5 ulangan; threshold dipilih
   dengan Youden's J; model sederhana dipilih bila selisih performa kecil.
9. **Penjelasan**: kontribusi tiap fitur dijumlah per tema -> kalimat alasan + tingkat lolos historis.
Sengaja tidak dipakai: jenis kelamin, kehamilan, tempat lahir, nama/KTP/HP/email, link CV, sumber lowongan, jawaban teks bebas
(diuji, tanpa sinyal), nama sekolah/jurusan/perusahaan.

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
| **Logistic Regression (dipakai)** | 0,706 | 0,630 | 0,622 | 0,67 | 0,58 |
| Random Forest | 0,709 | 0,644 | 0,586 | 0,67 | 0,52 |
| XGBoost | 0,700 | 0,643 | 0,566 | 0,67 | 0,49 |
| Baseline jarak saja | 0,608 | 0,512 | - | - | - |

Akurasi keseluruhan Logistic Regression (out-of-fold) 69%; naik ke 74% bila hanya memutuskan 42% pelamar yang paling pasti.
Fitur "tahu lowongan dari mana" **dihapus** (tidak relevan secara logika seleksi); AUC turun sedikit dari 0,721.
5-fold CV x 5 ulangan; threshold (0,56) dipilih dengan Youden's J. Logistic Regression dipilih karena performanya setara
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


## Data tambahan 2026 & form baru
- `form_baru/` : Google Form baru khusus Tele Marketing (header rapi, dropdown baku, kolom label untuk recruiter). Lihat `form_baru/PANDUAN.md`.
- `src/harmonize.py` + `build_training_data.py` : menyatukan 3 generasi form lama ke satu skema (lihat `docs/PENYAMAAN_HEADER.md`) dan membuat `laporan.md`.
- Temuan: pola 2024 (AUC 0,70 dengan fitur inti) TIDAK terulang di data 2026 (AUC 0,46-0,55; transfer antar periode ~0,5). Jangan menggabung
  2026 ke model sebelum proses seleksi 2026 dipahami dan label dari kolom `Status Seleksi Berkas` terkumpul.

## Training ulang
1. **Model yang sedang jalan (form lama / form baru):** `python train.py --data <jawaban.xlsx> --rekap Rekap_All.xlsx` (form lama, label dari rekap)
   atau `python train.py --data <jawaban form baru.xlsx> --all-dates` (label dari kolom "Status Seleksi Berkas"). Hasil: `models/model_interview.joblib`
   -> push ke GitHub (Streamlit dan Actions otomatis memakai yang baru).
2. **Eksperimen dari Excel gabungan:** `python train_unified.py --excel Data_Latih_Tele_Marketing.xlsx --skenario 2024|2026|semua`
   -> laporan AUC, akurasi vs baseline (tebakan kelas terbanyak), AUC dalam bulan yang sama, dan akurasi per cakupan keyakinan.
