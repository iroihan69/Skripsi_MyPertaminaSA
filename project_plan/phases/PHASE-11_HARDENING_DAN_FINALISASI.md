# Fase 11 - Hardening, Dokumentasi, dan Finalisasi

## Tujuan

Merapikan sistem agar reproducible, mudah dijalankan ulang, dan siap dipresentasikan atau ditulis pada bab implementasi.

## Dependensi

- Fase 10 selesai.

## Output Wajib

- README final proyek.
- Panduan run end-to-end.
- Daftar known issues dan batasan sistem.

## Checklist Masuk Fase

- [x] Pipeline utama sudah berjalan dari data sampai dashboard.
- [x] Model final sementara sudah dipilih.
- [x] Struktur file output final stabil.

## Langkah Implementasi

1. Rapikan README proyek.
2. Tulis urutan run dari scraping sampai dashboard.
3. Tambahkan known limitations sesuai hasil eksperimen.
4. Uji ulang pipeline minimal satu kali dari awal sampai akhir.
5. Catat kebutuhan demo offline bila diperlukan.
6. Pastikan semua artefak penting berada di lokasi yang benar.

## Checklist Validasi

- [x] Orang lain dapat mengikuti panduan run dasar.
- [x] Semua dependency yang benar terdokumentasi.
- [x] Batasan sistem dijelaskan jujur.
- [x] Artefak demo tersedia.

## Gate Pindah Fase

- [x] Sistem siap diuji atau dipresentasikan.
- [x] Dokumen run akhir lengkap.
- [x] Tidak ada blocker kritis yang belum dicatat.

## Risiko Fase

- Dokumentasi tertinggal dari implementasi.
- Dependensi lokal tidak sama dengan lingkungan baru.

## Catatan Eksekusi

- Tanggal eksekusi: 2026-03-24.
- Strategi validasi: smoke test end-to-end mode demo offline (tanpa scraping live).
- Validasi pipeline yang dijalankan:
	- `python -m src.preprocessing.qa_raw_data`
	- `python -m src.preprocessing.preprocess_reviews --stopword-mode selective`
	- `python -m src.preprocessing.label_and_split --test-size 0.2`
	- `python -m src.modeling.evaluate_indobert --model-dir model_output/baseline_indobert`
	- `python -m src.modeling.predict_indobert --model-dir model_output/baseline_indobert --input data/processed/test_data.csv --output data/predictions/predictions.csv`
	- `python -m streamlit run dashboard/app.py --server.headless true --server.port 8511`
- Ringkasan hasil validasi:
	- QA raw data: decision "Lanjut ke preprocessing", flagged rows=0.
	- Preprocessing: input 5000, output 4964, dropped 36.
	- Label split: train 3971, test 993.
	- Evaluasi: weighted F1 0.8322, baseline diterima.
	- Inferensi: output 993 baris, confidence valid pada rentang 0-1.
	- Dashboard: startup headless berhasil (smoke startup lulus).
- Dokumentasi finalisasi:
	- README proyek dirapikan dengan panduan setup Windows (termasuk execution policy), run end-to-end, known issues, dan kebutuhan demo offline.
	- Daftar artefak output penting dipastikan konsisten dengan struktur folder final.