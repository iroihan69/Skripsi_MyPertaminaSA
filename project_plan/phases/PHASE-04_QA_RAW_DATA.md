# Fase 04 - Quality Check Raw Data

## Tujuan

Memastikan data mentah layak masuk ke tahap preprocessing dan tidak membawa masalah mendasar ke fase berikutnya.

## Dependensi

- Fase 03 selesai.

## Output Wajib

- Ringkasan kualitas data mentah.
- Daftar isu raw data jika ada.
- Keputusan data siap preprocess atau perlu scraping ulang.

## Checklist Masuk Fase

- [x] `raw_reviews.csv` tersedia.
- [x] Jumlah data minimum untuk eksperimen sudah tercapai (`5000` baris, mengikuti target minimum Fase 03).

## Langkah Implementasi

1. Cek jumlah total baris.
2. Cek nilai null pada kolom wajib.
3. Cek duplikasi berdasarkan `reviewId` dan kombinasi konten-tanggal.
4. Cek distribusi score 1-5.
5. Cek kualitas tanggal pada kolom `at`.
6. Buat ringkasan awal noise data seperti emoji berat, URL, atau teks kosong.

## Checklist Validasi

- [x] Kolom wajib tidak hilang.
- [x] Nilai null pada `content` dan `score` sudah diketahui.
- [x] Distribusi rating masuk akal.
- [x] Aturan penanganan data rusak disepakati.

## Gate Pindah Fase

- [x] Ada keputusan final: lanjut preprocessing atau ulang scraping.
- [x] Data yang tidak valid sudah dibuang atau ditandai.
- [x] Ringkasan kualitas data terdokumentasi.

## Catatan Eksekusi 2026-03-16

- Script QA: `src/preprocessing/qa_raw_data.py`
- Command: `python -m src.preprocessing.qa_raw_data`
- Input: `data/raw/raw_reviews.csv`
- Output ringkasan: `logs/raw_data_qa_report.md` dan `logs/raw_data_qa_summary.json`
- Output penandaan isu: `data/raw/raw_reviews_qa_flags.csv`
- Hasil validasi utama: 5000 baris, kolom wajib lengkap, null `content`/`score`/`at` = 0, duplikasi `reviewId` = 0, duplikasi `content-at` = 0, tanggal invalid = 0.
- Distribusi score: 1=1768 (35,36%), 2=184 (3,68%), 3=237 (4,74%), 4=291 (5,82%), 5=2520 (50,40%).
- Noise awal: URL = 0, emoji berat = 62, simbol berat = 140.
- Keputusan fase: data layak lanjut ke preprocessing.
- Aturan penanganan data rusak: pertahankan `raw_reviews.csv` apa adanya; gunakan `raw_reviews_qa_flags.csv` untuk menandai baris bermasalah sebelum dikeluarkan pada fase preprocessing. Pada eksekusi ini tidak ada baris yang ter-flag.

## Risiko Fase

- Banyak data kosong.
- Distribusi rating terlalu timpang.
- Data tanggal tidak konsisten.