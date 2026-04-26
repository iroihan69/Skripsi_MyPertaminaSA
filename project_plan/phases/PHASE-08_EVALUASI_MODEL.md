# Fase 08 - Evaluasi Model dan Error Analysis

## Tujuan

Mengukur performa model dan menentukan apakah model layak dipakai, perlu tuning, atau perlu revisi data.

## Dependensi

- Fase 07 selesai.

## Output Wajib

- Accuracy, precision, recall, weighted F1.
- Confusion matrix.
- Ringkasan error analysis.

## Checklist Masuk Fase

- [x] Model terbaik tersedia.
- [x] Test set final tersedia.
- [x] Metrik evaluasi yang dipakai sudah disepakati.

## Langkah Implementasi

1. Jalankan evaluasi pada test set.
2. Hitung confusion matrix.
3. Buat classification report.
4. Identifikasi kelas yang paling sering tertukar.
5. Ambil sampel kesalahan prediksi untuk inspeksi manual.
6. Bandingkan hasil dengan target performa pada TRD.

## Checklist Validasi

- [x] Semua metrik utama tersedia.
- [x] Confusion matrix bisa dibaca.
- [x] Ada analisis mengapa error dominan terjadi.
- [x] Keputusan lanjut atau revisi dapat dipertanggungjawabkan.

## Gate Pindah Fase

- [x] Ada keputusan final baseline: terima, tuning ulang, atau revisi data.
- [x] Hasil evaluasi terdokumentasi.
- [x] Risiko kualitas model untuk dashboard sudah dipahami.

## Risiko Fase

- Metrik tinggi tetapi bias pada kelas mayoritas.
- Label netral sangat ambigu.

## Catatan Eksekusi

- Tanggal eksekusi: 2026-03-24.
- Script evaluasi: `python -m src.modeling.evaluate_indobert`.
- Artefak laporan: `logs/evaluation_report.md` dan `logs/evaluation_summary.json`.
- Artefak confusion matrix: `logs/evaluation_confusion_matrix.png`.
- Artefak sampel error: `logs/evaluation_misclassified_samples.csv`.
- Ringkasan metrik: accuracy 0.8520, weighted precision 0.8151, weighted recall 0.8520, weighted F1 0.8322.
- Decision rule yang disepakati: baseline lulus jika weighted F1 >= 0.80.
- Keputusan baseline: terima baseline dan lanjut Fase 09.
- Risiko residu utama: kelas Netral belum terdeteksi (recall/F1 = 0.0000), sehingga perlu perhatian pada Fase 09-11 agar bias kelas tidak menyesatkan dashboard.