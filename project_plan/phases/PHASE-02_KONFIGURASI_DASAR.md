# Fase 02 - Konfigurasi Dasar dan Utilitas Proyek

## Tujuan

Menyiapkan konfigurasi dan utilitas dasar agar seluruh modul memakai pola yang konsisten.

## Dependensi

- Fase 01 selesai.

## Output Wajib

- Modul konfigurasi path dan konstanta.
- Helper umum untuk I/O file, logging, dan validasi sederhana.
- Penamaan file output yang konsisten.

## Checklist Masuk Fase

- [x] Struktur folder final sudah tersedia.
- [x] Dependency Python dasar sudah terpasang.
- [x] Penamaan direktori `data` dan `model_output` tidak akan diubah lagi.

## Langkah Implementasi

1. Buat file konfigurasi sentral untuk path proyek dan nama file utama.
2. Definisikan konstanta seperti `APP_ID`, `LANG`, `COUNTRY`, `MODEL_NAME`, dan seed.
3. Buat helper penyimpanan CSV dan pembuatan folder otomatis.
4. Buat logging sederhana untuk scraping, preprocessing, training, dan dashboard.
5. Tentukan pola penamaan file hasil proses agar mudah dilacak.

## Checklist Validasi

- [x] Semua path relatif bekerja dari root project.
- [x] Konstanta inti tidak tersebar di banyak file.
- [x] Utilitas dasar dapat dipakai ulang oleh modul lain.

## Gate Pindah Fase

- [x] Ada satu sumber konfigurasi utama.
- [x] Seluruh folder output dibuat otomatis jika belum ada.
- [x] Logging dasar tersedia untuk proses ETL dan training.

## Risiko Fase

- Konfigurasi hard-coded.
- Helper terlalu spesifik dan sulit dipakai ulang.

## Catatan Implementasi

- Konfigurasi inti dipusatkan di `src/config.py` mencakup konstanta, path proyek, nama file output, dan pemetaan output per stage.
- Utilitas reusable ditambahkan pada `src/utils/io_utils.py`, `src/utils/logging_utils.py`, `src/utils/validation_utils.py`, dan `src/utils/naming_utils.py`.
- Ekspor utilitas disatukan lewat `src/utils/__init__.py` agar impor antar modul lebih konsisten.