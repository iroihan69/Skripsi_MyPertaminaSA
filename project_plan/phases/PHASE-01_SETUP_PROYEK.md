# Fase 01 - Setup Proyek dan Environment

## Tujuan

Membuat pondasi proyek yang rapi agar fase data, modeling, dan dashboard tidak berjalan di struktur yang berubah-ubah.

## Dependensi

- TRD utama tersedia.
- Keputusan bahasa implementasi adalah Python.

## Output Wajib

- Struktur folder proyek awal sesuai TRD appendix.
- `requirements.txt` awal.
- Virtual environment atau environment manager yang dipilih.
- `README.md` minimum untuk setup lokal.

## Checklist Masuk Fase

- [x] Nama folder proyek final sudah disepakati — `mypertamina-sentiment/`.
- [x] Versi Python target ditetapkan — Python 3.13.3.
- [x] Sistem operasi utama pengembangan diketahui — Windows 11.
- [x] Lokasi penyimpanan data dan model sudah dipilih — dalam folder proyek (`d:\File Opieq\MypertaminaSA\mypertamina-sentiment\`).

## Langkah Implementasi

1. Buat struktur direktori utama: `data/`, `src/`, `dashboard/`, `model_output/`, `notebooks/`.
2. Buat subfolder data: `data/raw/`, `data/processed/`, `data/predictions/`.
3. Buat subfolder source: `src/scraping/`, `src/preprocessing/`, `src/modeling/`, `src/utils/`.
4. Tentukan manajemen environment: `venv`, `conda`, atau lain-lain.
5. Susun `requirements.txt` dari dependency pada TRD.
6. Tambahkan `.gitignore` yang sesuai untuk Python, model output, cache, dan environment lokal.
7. Buat `README.md` minimal berisi cara setup dan run dasar.

## Checklist Validasi

- [x] Struktur folder sesuai appendix TRD.
- [x] Dependency inti sudah tercantum di `requirements.txt`.
- [ ] Environment bisa diaktifkan tanpa konflik — *perlu diverifikasi setelah `pip install`*.
- [x] Tidak ada file sensitif atau file cache yang ikut dalam versioning (dicakup `.gitignore`).

## Gate Pindah Fase

Semua item berikut wajib selesai sebelum masuk Fase 02:

- [x] Proyek bisa dibuka dan dijalankan lokal tanpa error setup dasar — terverifikasi: torch 2.10.0, transformers 5.3.0, pandas 2.3.3 berjalan di `.venv`.
- [x] `requirements.txt` sudah dapat dipakai untuk instalasi awal.
- [x] Struktur folder final tidak berubah-ubah lagi.
- [x] `README.md` setup minimum sudah ada.

> **Catatan Eksekusi:** Selesai 2026-03-16. Environment manager: `venv` (Python 3.13.3). Seluruh gate terpenuhi. Lihat REV-005 untuk catatan fix execution policy Windows.

## Risiko Fase

- Versi package saling konflik.
- Struktur folder berubah di tengah implementasi.
- Path absolut dipakai terlalu awal.