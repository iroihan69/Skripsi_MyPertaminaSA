# Roadmap Implementasi Sistem Analisis Sentimen MyPertamina

## Tujuan

Roadmap ini memecah TRD menjadi fase-fase kecil yang bisa dikerjakan bertahap, dengan checkpoint yang jelas sebelum lanjut ke fase berikutnya.

## Ringkasan Fase

| Fase | Nama | Fokus Utama | Output Minimum | Lanjut Jika |
|---|---|---|---|---|
| 01 | Setup Proyek | Struktur repo, environment, dependency awal | Struktur folder dan `requirements.txt` | Project bisa dijalankan lokal |
| 02 | Konfigurasi Dasar | Config, helper, penamaan path, logging | File konfigurasi dan utilitas dasar | Semua path dan parameter inti tervalidasi |
| 03 | Scraping Data | Pengambilan ulasan Google Play | `raw_reviews.csv` awal | Data berhasil diambil dan disimpan |
| 04 | QA Raw Data | Validasi kualitas data mentah | Laporan cek data mentah | Kolom wajib lengkap dan usable |
| 05 | Preprocessing Teks | Cleaning, normalisasi, pipeline transform | `preprocessed_reviews.csv` | Teks bersih dan konsisten |
| 06 | Labeling dan Split | Mapping label dan pembagian data | `train_data.csv` dan `test_data.csv` | Distribusi kelas valid |
| 07 | Fine-tuning IndoBERT | Training model klasifikasi | Folder `model_output/` | Training selesai dan model tersimpan |
| 08 | Evaluasi Model | Metric, confusion matrix, error analysis | Hasil evaluasi model | Metrik minimum terpenuhi atau ada keputusan revisi |
| 09 | Inferensi Batch | Prediksi data dan penyimpanan hasil | `predictions.csv` | Format output siap dibaca dashboard |
| 10 | Dashboard Streamlit | Visualisasi dan filter analisis | Aplikasi dashboard berjalan | Semua komponen dashboard tampil benar |
| 11 | Hardening dan Finalisasi | Testing akhir, dokumentasi, reproducibility | README, panduan run, cek akhir | Sistem siap demo atau penulisan bab implementasi |

## Prinsip Eksekusi

- Satu fase memiliki satu tujuan utama dan output yang dapat diverifikasi.
- Jika fase gagal di gate, perbaikannya kembali ke fase itu atau fase sebelumnya.
- Semua perubahan di luar scope fase dicatat di `BACKLOG_REVISI.md`.

## Jalur Eksekusi Utama

1. Fase 01-02 membangun pondasi proyek.
2. Fase 03-06 menyiapkan data yang siap latih.
3. Fase 07-09 menghasilkan model dan output prediksi.
4. Fase 10-11 menyelesaikan lapisan presentasi dan finalisasi sistem.

## Fase Aktif Pertama

Mulai dari `phases/PHASE-01_SETUP_PROYEK.md` karena Anda memilih fokus awal pada setup project.