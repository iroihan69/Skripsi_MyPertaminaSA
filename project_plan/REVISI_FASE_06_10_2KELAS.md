# Revisi Rerun Fase 06-10 - Skema 2 Kelas

## Tujuan

Dokumen ini dipakai sebagai catatan revisi untuk rerun pipeline Fase 06 sampai Fase 10 dengan fokus skema 2 kelas, class weighting, dan evaluasi yang lebih adil terhadap distribusi data.

## Keputusan Utama

- Gunakan skema label 2 kelas sebagai model utama penelitian.
- Gunakan class weighting pada training untuk menahan bias kelas negatif yang dominan.
- Jadikan 3 kelas sebagai temuan eksploratif, bukan model final, jika performa netral tetap lemah.
- Jangan pakai dataset yang sama lagi untuk inferensi produksi.

## Catatan Penting Sebelum Rerun

- Distribusi data 2 kelas saat ini berat ke negatif.
- Fokus evaluasi utama: macro F1, recall per kelas, precision per kelas, dan confusion matrix.
- Accuracy tidak boleh jadi metrik utama.
- Semua split harus stratified.
- Inferensi hanya untuk data unseen.

## Reminder Running per Fase

### Fase 06 - Labeling dan Split

- Konfirmasi dulu sebelum run: apakah skema yang dipakai hanya 2 kelas.
- Pakai mapping 2 kelas: rating 1-3 = negatif, rating 4-5 = positif.
- Gunakan stratified split.
- Simpan catatan distribusi label train dan test.
- Jangan campur data inferensi ke train/test.

Status konfirmasi: [x] Setuju / [ ] Revisi

### Fase 07 - Fine-tuning IndoBERT

- Konfirmasi dulu sebelum run: class weighting aktif.
- Gunakan bobot kelas dari distribusi train set.
- Default rumus: w_i = N / (K * n_i).
- Jika training terlalu tidak stabil, gunakan bobot yang di-smooth.
- Metric utama model terbaik: macro_f1.

Status konfirmasi: [ ] Setuju / [ ] Revisi

### Fase 08 - Evaluasi Model

- Konfirmasi dulu sebelum run: evaluasi wajib tampilkan per-class report.
- Laporkan macro F1, weighted F1, precision, recall, dan confusion matrix.
- Cek apakah recall kelas positif membaik setelah class weighting.
- Jika 3 kelas diuji, jadikan hasilnya sebagai pembanding, bukan model final.

Status konfirmasi: [x] Setuju / [ ] Revisi

### Fase 09 - Inferensi Batch

- Konfirmasi dulu sebelum run: data inferensi harus unseen.
- Jangan pakai ulang train/test untuk prediksi produksi.
- Simpan hasil prediksi terpisah dari dataset pelatihan.
- Pastikan output inferensi bisa dibedakan dari data evaluasi.

Status konfirmasi: [x] Setuju / [ ] Revisi

### Fase 10 - Dashboard Streamlit

- Konfirmasi dulu sebelum run: dashboard menampilkan keterbatasan data.
- Tampilkan distribusi label, macro F1, dan recall per kelas.
- Jangan hanya tampilkan accuracy atau proporsi prediksi.
- Jika 3 kelas gagal stabil, tampilkan sebagai limitation penelitian.

Status konfirmasi: [x] Setuju / [ ] Revisi

## Catatan Interpretasi Penelitian

- Model 2 kelas lebih cocok untuk dataset ini karena distribusinya lebih realistis untuk ditangani dengan class weighting.
- Model 3 kelas dapat tetap dicoba, tetapi kemungkinan besar netral akan sulit stabil karena jumlahnya kecil.
- Jika 3 kelas tidak layak, itu boleh dijadikan temuan penelitian yang sah.
- Penelitian tetap valid selama alasan teknisnya dijelaskan dengan data dan metrik yang tepat.

## Checklist Final Sebelum Eksekusi

- [x] Skema label yang dipakai sudah diputuskan.
- [x] Class weighting sudah disepakati.
- [x] Target metrik utama sudah ditetapkan.
- [x] Dataset inferensi sudah dipisahkan.
- [x] Catatan limitation untuk 3 kelas sudah siap.

## Catatan Eksekusi Fase 06 (2026-04-21)

- Skema label dipakai: 2 kelas (rating 1-3 negatif, rating 4-5 positif).
- Split bertahap stratified:
	- Tahap 1: 10% inference holdout dari total data.
	- Tahap 2: dari sisa 90%, dibagi train 80% dan test 20%.
- Validasi anti-kebocoran reviewId: train-test = 0, train-holdout = 0, test-holdout = 0.
- Output baru (tanpa overwrite file lama):
	- data/processed/train_data_revisi_2kelas_20260421.csv
	- data/processed/test_data_revisi_2kelas_20260421.csv
	- data/processed/inference_holdout_revisi_2kelas_20260421.csv
	- logs/label_split_summary_revisi_2kelas_20260421.json
	- logs/label_split_report_revisi_2kelas_20260421.md

## Catatan Perubahan di Luar Fase 06

- Tidak ada perubahan di luar ruang lingkup Fase 06 pada eksekusi ini.

## Catatan Eksekusi Fase 07 (2026-04-21)

- Skema label dipakai: 2 kelas (Negatif/Positif).
- Konfigurasi utama disepakati sebelum run:
	- Class weighting: aktif (rumus w_i = N / (K * n_i)).
	- Metric pemilihan model terbaik: macro_f1.
	- Hyperparameter modifikasi: learning_rate=2e-5, epochs=3, train_batch_size=16, max_length=128.
- Output baru (tanpa overwrite file lama):
	- model_output/baseline_indobert_revisi_2kelas/
	- logs/training_revisi_2kelas_20260421.log
	- logs/training_summary_revisi_2kelas_20260421.json
	- logs/training_report_revisi_2kelas_20260421.md
- Ringkasan hasil training:
	- Device: CPU.
	- Runtime training: 5242.26 detik.
	- Best checkpoint: model_output/baseline_indobert_revisi_2kelas/checkpoint-350.
	- Best metric (macro_f1): 0.8702.
	- Evaluasi akhir: weighted_f1=0.9318, macro_f1=0.8702, accuracy=0.9334.
	- Bobot kelas train: Negatif=0.5953, Positif=3.1240.

## Catatan Perubahan di Luar Fase 07

- Tidak ada perubahan di luar ruang lingkup Fase 07 pada eksekusi ini.

## Catatan Eksekusi Fase 08 (2026-04-21)

- Skema evaluasi dipakai: 2 kelas (Negatif/Positif).
- Konfigurasi evaluasi disepakati sebelum run:
	- Model: model_output/baseline_indobert_revisi_2kelas
	- Dataset evaluasi: data/processed/test_data_revisi_2kelas_20260421.csv
	- Decision threshold weighted F1: 0.80
	- Label scheme: 2class
- Output baru (tanpa overwrite file lama):
	- logs/evaluation_report_revisi_2kelas_20260421.md
	- logs/evaluation_summary_revisi_2kelas_20260421.json
	- logs/evaluation_confusion_matrix_revisi_2kelas_20260421.png
	- logs/evaluation_misclassified_samples_revisi_2kelas_20260421.csv
- Ringkasan hasil evaluasi:
	- Jumlah data evaluasi: 24529 baris.
	- Accuracy: 0.9334.
	- Weighted precision: 0.9312.
	- Weighted recall: 0.9334.
	- Weighted F1: 0.9318.
	- Macro F1: 0.8702.
	- Recall per kelas: Negatif=0.9709, Positif=0.7364.
	- Confusion matrix utama: Aktual Negatif->Prediksi Positif=599, Aktual Positif->Prediksi Negatif=1035.
	- Total misclassified: 1634.
	- Keputusan baseline: Terima baseline dan lanjut Fase 09.

## Catatan Perubahan di Luar Fase 08

- Tidak ada perubahan di luar ruang lingkup Fase 08 pada eksekusi ini.

## Catatan Eksekusi Fase 09 (2026-04-21)

- Skema inferensi dipakai: 2 kelas (Negatif/Positif).
- Konfigurasi inferensi disepakati sebelum run:
	- Model: model_output/baseline_indobert_revisi_2kelas
	- Dataset inferensi (unseen holdout): data/processed/inference_holdout_revisi_2kelas_20260421.csv
	- Label scheme: 2class
	- Batch size: 32
	- Max length: 128
- Output baru (tanpa overwrite file lama):
	- data/predictions/predictions_revisi_2kelas_20260421.csv
	- logs/inference_summary_revisi_2kelas_20260421.json
	- logs/inference_report_revisi_2kelas_20260421.md
- Ringkasan hasil inferensi:
	- Jumlah data inferensi: 13627 baris.
	- Jumlah output prediksi: 13627 baris.
	- Kolom `at` tetap terbawa: Ya.
	- Distribusi prediksi: Negatif=11699, Positif=1928.
	- Rentang confidence: min=0.5001, max=0.9931, mean=0.9423.

## Catatan Perubahan di Luar Fase 09

- Tidak ada perubahan di luar ruang lingkup Fase 09 pada eksekusi ini.

## Catatan Eksekusi Fase 10 (2026-04-21)

- Skema dashboard dipakai: 2 kelas (Negatif/Positif) sesuai baseline revisi.
- Konfigurasi Fase 10 disepakati sebelum run:
	- Dashboard dibuat sebagai file baru tanpa mengubah dashboard existing.
	- Nama file dashboard baru: `dashboard/app_revisi.py`.
	- Dataset default dashboard: `data/predictions/predictions_revisi_2kelas_20260421.csv`.
	- Summary evaluasi default: `logs/evaluation_summary_revisi_2kelas_20260421.json`.
- Fitur tambahan yang diimplementasikan:
	- Bar chart penyebaran sentimen Positif dan Negatif per tahun.
	- Panel metrik evaluasi model berisi Macro F1, Weighted F1, Recall Negatif, Recall Positif.
	- Catatan limitation untuk skema 3 kelas ditampilkan pada dashboard.
- Output baru (tanpa overwrite file lama):
	- mypertamina-sentiment/dashboard/app_revisi.py
- Validasi teknis:
	- Sintaks script valid (`python -m py_compile dashboard/app_revisi.py`).
	- Startup Streamlit berhasil (`python -m streamlit run dashboard/app_revisi.py --server.headless true --server.port 8511`).

## Catatan Perubahan di Luar Fase 10

- Tidak ada perubahan di luar ruang lingkup Fase 10 pada eksekusi ini.
