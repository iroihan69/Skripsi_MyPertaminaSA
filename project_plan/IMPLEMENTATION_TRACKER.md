# Implementation Tracker

## Cara Pakai

- Update status per fase secara manual.
- Checklist hanya boleh ditandai selesai jika output dan gate fase benar-benar terpenuhi.
- Jika ada blocker, tulis ringkas pada kolom `Catatan` dan rincian lanjut di `BACKLOG_REVISI.md`.

## Status Global

| Fase | Status | Output Utama | Catatan |
|---|---|---|---|
| 01. Setup Proyek | Selesai | Struktur repo dan dependency dasar | Struktur folder, `requirements.txt`, dan inisialisasi package sudah tersedia |
| 02. Konfigurasi Dasar | Selesai | Config, helper, logging | Konfigurasi terpusat + utilitas I/O, validasi, logging, naming selesai |
| 03. Scraping Data | Selesai | `raw_reviews.csv` | Scraping live selesai (5000 baris, kolom wajib lengkap, duplikasi reviewId=0); lihat catatan APP_ID pada `REV-007` |
| 04. QA Raw Data | Belum mulai | Validasi data mentah |  |
| 05. Preprocessing Teks | Belum mulai | `preprocessed_reviews.csv` |  |
| 06. Labeling dan Split | Belum mulai | `train_data.csv`, `test_data.csv` |  |
| 07. Fine-tuning IndoBERT | Belum mulai | `model_output/` |  |
| 08. Evaluasi Model | Belum mulai | Metric dan analisis error |  |
| 09. Inferensi Batch | Belum mulai | `predictions.csv` |  |
| 10. Dashboard Streamlit | Belum mulai | Dashboard siap run |  |
| 11. Hardening dan Finalisasi | Belum mulai | Dokumen run dan cek akhir |  |

## Checklist Master

- [x] Fase 01 selesai
- [x] Fase 02 selesai
- [x] Fase 03 selesai
- [ ] Fase 04 selesai
- [ ] Fase 05 selesai
- [ ] Fase 06 selesai
- [ ] Fase 07 selesai
- [ ] Fase 08 selesai
- [ ] Fase 09 selesai
- [ ] Fase 10 selesai
- [ ] Fase 11 selesai

## Definisi Status

- `Belum mulai` : belum ada pengerjaan.
- `Berjalan` : implementasi sedang aktif.
- `Tertahan` : ada blocker atau keputusan belum final.
- `Selesai` : semua gate fase lulus.