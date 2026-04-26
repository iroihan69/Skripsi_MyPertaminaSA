# Fase 07 - Fine-tuning IndoBERT

## Tujuan

Melatih model IndoBERT untuk klasifikasi sentimen tiga kelas berdasarkan dataset yang sudah disiapkan.

## Dependensi

- Fase 06 selesai.

## Output Wajib

- Script training.
- Model hasil fine-tuning di `model_output/`.
- Log training dan parameter yang dipakai.

## Checklist Masuk Fase

- [x] Dataset train dan test tersedia.
- [x] `transformers` dan `torch` siap dipakai.
- [x] `MODEL_NAME` dipilih final.
- [x] Hyperparameter baseline ditetapkan.

## Langkah Implementasi

1. Muat tokenizer dan model `indobenchmark/indobert-base-p1`.
2. Siapkan dataset tokenized dengan `max_length` yang dipilih.
3. Implementasikan `TrainingArguments` dan `Trainer`.
4. Tentukan metrik evaluasi utama, minimal weighted F1.
5. Jalankan training baseline.
6. Simpan model terbaik dan tokenizer.

## Checklist Validasi

- [x] Training berjalan sampai selesai.
- [x] Model terbaik tersimpan.
- [x] Tidak ada error fatal memori atau shape tensor.
- [x] Seed dan hyperparameter tercatat.

## Gate Pindah Fase

- [x] Ada model hasil training yang dapat dimuat ulang.
- [x] Ada log training yang dapat dijelaskan.
- [x] Konfigurasi training baseline sudah dibekukan.

## Hasil Eksekusi Baseline

- Tanggal eksekusi: 2026-03-17.
- Model: `indobenchmark/indobert-base-p1`.
- Device: CPU.
- Artefak model: `mypertamina-sentiment/model_output/baseline_indobert/`.
- Ringkasan metrik: weighted F1 = 0.8322, accuracy = 0.8520.
- Log dan laporan:
	- `mypertamina-sentiment/logs/training.log`
	- `mypertamina-sentiment/logs/training_summary.json`
	- `mypertamina-sentiment/logs/training_report.md`

## Risiko Fase

- GPU tidak cukup.
- Overfitting cepat.
- Waktu training terlalu lama di CPU.