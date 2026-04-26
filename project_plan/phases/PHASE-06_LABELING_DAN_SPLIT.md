# Fase 06 - Labeling Sentimen dan Dataset Split

## Tujuan

Membentuk dataset supervised siap latih dengan label sentimen yang konsisten dan pembagian train-test yang valid.

## Dependensi

- Fase 05 selesai.

## Output Wajib

- Kolom `label`.
- `train_data.csv`.
- `test_data.csv`.
- Ringkasan distribusi kelas.

## Checklist Masuk Fase

- [x] Kolom `score` valid.
- [x] Kolom `content_clean` siap dipakai.
- [x] Aturan labeling 3 kelas sudah final.

## Langkah Implementasi

1. Mapping rating 1-2 menjadi `Negatif=0`.
2. Mapping rating 3 menjadi `Netral=1`.
3. Mapping rating 4-5 menjadi `Positif=2`.
4. Validasi distribusi label hasil mapping.
5. Lakukan stratified train-test split 80:20.
6. Simpan hasil split ke `data/processed/`.

## Checklist Validasi

- [x] Semua baris punya label numerik yang valid.
- [x] Distribusi label train dan test proporsional.
- [x] Tidak ada kebocoran data yang jelas antar split.

## Gate Pindah Fase

- [x] `train_data.csv` dan `test_data.csv` tersedia.
- [x] Distribusi kelas terdokumentasi.
- [x] Keputusan penanganan imbalance awal dibuat.

## Catatan Eksekusi 2026-03-17

- Script labeling/split: `src/preprocessing/label_and_split.py`
- Command: `python -m src.preprocessing.label_and_split`
- Input: `data/processed/preprocessed_reviews.csv`
- Output utama: `data/processed/train_data.csv` (3971 baris) dan `data/processed/test_data.csv` (993 baris)
- Output ringkasan: `logs/label_split_summary.json` dan `logs/label_split_report.md`
- Aturan labeling final 3 kelas: rating 1-2 -> `Negatif=0`, rating 3 -> `Netral=1`, rating 4-5 -> `Positif=2`.
- Distribusi label keseluruhan: Negatif 1952 (39,32%), Netral 237 (4,77%), Positif 2775 (55,90%).
- Distribusi train dan test proporsional melalui stratified split 80:20.
- Validasi split: overlap `reviewId` antara train-test = 0, sehingga tidak ada indikasi kebocoran data yang jelas.
- Keputusan awal imbalance: baseline Fase 07 dijalankan tanpa rebalancing; pertimbangkan class weighting jika metrik kelas minoritas rendah.

## Risiko Fase

- Class imbalance berat.
- Label rating tidak selalu merepresentasikan sentimen teks.