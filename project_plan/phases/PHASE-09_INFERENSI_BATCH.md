# Fase 09 - Inferensi Batch dan Penyimpanan Prediksi

## Tujuan

Menyiapkan pipeline prediksi yang membaca teks, menjalankan model, dan menyimpan hasil dalam format siap dashboard.

## Dependensi

- Fase 08 selesai atau minimal model baseline sudah diterima sementara.

## Output Wajib

- Script prediksi batch.
- `predictions.csv`.
- Format confidence score dan label name yang konsisten.

## Checklist Masuk Fase

- [x] Model final sementara tersedia.
- [x] Tokenizer final tersedia.
- [x] Skema output prediksi sudah disetujui.

## Langkah Implementasi

1. Implementasikan fungsi `predict_sentiment(texts, model_path)`.
2. Keluarkan `predicted_label`, `label_name`, dan `confidence`.
3. Pastikan kolom tanggal `at` tetap ikut jika sumber data memilikinya.
4. Simpan hasil ke `data/predictions/predictions.csv`.
5. Validasi konsistensi tipe data untuk dibaca dashboard.

## Checklist Validasi

- [x] File prediksi dapat dibuka ulang.
- [x] Nilai confidence berada pada rentang valid.
- [x] Label numerik dan label teks sinkron.
- [x] Tidak ada baris yang hilang tanpa alasan jelas.

## Gate Pindah Fase

- [x] `predictions.csv` siap dipakai dashboard tanpa transform tambahan besar.
- [x] Fungsi prediksi dapat dipakai ulang.
- [x] Format output sudah stabil.

## Risiko Fase

- Mapping label inkonsisten dengan fase training.
- Confidence score tidak tervalidasi.

## Catatan Eksekusi

- Tanggal eksekusi: 2026-03-24.
- Script inferensi batch: `python -m src.modeling.predict_indobert --input data/processed/test_data.csv --output data/predictions/predictions.csv`.
- Fungsi reusable tersedia di `src/modeling/predict_indobert.py` melalui `predict_sentiment(texts, model_path)`.
- Artefak utama: `data/predictions/predictions.csv`.
- Artefak validasi: `logs/inference_summary.json` dan `logs/inference_report.md`.
- Ringkasan validasi output:
	- Jumlah baris input = 993 dan output = 993 (tidak ada baris hilang).
	- Kolom wajib inferensi tersedia: `predicted_label`, `label_name`, `confidence`.
	- Kolom `at` tetap terbawa dari sumber input.
	- Rentang confidence valid: min 0.3602, max 0.9857.
- Catatan residual kualitas model: prediksi label Netral masih 0 pada data uji (konsisten dengan risiko evaluasi Fase 08), sehingga perlu perhatian saat desain insight dashboard pada Fase 10.