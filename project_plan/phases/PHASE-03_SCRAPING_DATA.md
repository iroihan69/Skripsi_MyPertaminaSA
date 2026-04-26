# Fase 03 - Scraping Data Ulasan

## Tujuan

Mengambil data ulasan MyPertamina dari Google Play Store dan menyimpannya sebagai dataset mentah yang konsisten.

## Dependensi

- Fase 01 selesai.
- Fase 02 minimal konfigurasi dasar untuk path dan parameter scraping tersedia.

## Output Wajib

- Script scraping.
- File `raw_reviews.csv`.
- Parameter scraping terdokumentasi.

## Checklist Masuk Fase

- [x] `google-play-scraper` tersedia.
- [x] Keputusan APP_ID implementasi: `com.dafturn.mypertamina` (lihat `REV-007`).
- [x] Count awal atau strategi pagination sudah ditentukan (`count=5000`, `batch_size=200`).
- [x] Strategi penyimpanan raw data sudah diputuskan (`data/raw/raw_reviews.csv`).

## Langkah Implementasi

1. Implementasikan fungsi `scrape_reviews(...)` sesuai kontrak TRD.
2. Ambil field minimum: `reviewId`, `content`, `score`, `at`, `userName`, `thumbsUpCount`.
3. Tambahkan metadata opsional seperti waktu scraping dan jumlah data.
4. Simpan hasil ke `data/raw/raw_reviews.csv`.
5. Tangani error koneksi, rate limit, atau respons kosong.
6. Jika perlu, tambahkan delay atau batching.

## Checklist Validasi

- [x] Data berhasil diambil dari Google Play Store.
- [x] File CSV dapat dibuka dan terbaca oleh Pandas.
- [x] Kolom wajib sesuai TRD.
- [x] Tidak ada duplikasi besar yang jelas akibat scraping berulang.

## Gate Pindah Fase

- [x] `raw_reviews.csv` tersedia di lokasi final.
- [x] Jumlah baris memenuhi target minimum yang Anda tetapkan (`5000`).
- [x] Script scraping dapat dijalankan ulang dengan hasil yang konsisten.

## Catatan Eksekusi 2026-03-16

- Script: `src/scraping/scraper.py`
- Command: `python -m src.scraping.scraper --count 5000 --start-date 2022-01-01 --end-date 2025-12-31 --app-id com.dafturn.mypertamina`
- Output: `data/raw/raw_reviews.csv`
- Hasil validasi: 5000 baris, kolom wajib lengkap, duplikasi `reviewId` = 0.

## Risiko Fase

- Endpoint berubah atau dibatasi.
- Format data berbeda dari ekspektasi.
- Data sangat sedikit atau banyak duplikasi.