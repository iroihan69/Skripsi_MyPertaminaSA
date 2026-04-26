# Fase 05 - Preprocessing Teks Bahasa Indonesia

## Tujuan

Membersihkan dan menormalkan teks ulasan agar siap dipakai untuk labeling dan tokenisasi model.

## Dependensi

- Fase 04 selesai.

## Output Wajib

- Script preprocessing.
- Kolom `content_clean`.
- File `preprocessed_reviews.csv` awal.

## Checklist Masuk Fase

- [x] Aturan cleaning dasar sudah disetujui.
- [x] Keputusan normalisasi slang dibuat, minimal daftar awal.
- [x] Keputusan stopword removal sudah jelas: dipakai atau tidak.

## Langkah Implementasi

1. Lakukan case folding.
2. Hapus URL, angka, tanda baca, emoji, dan karakter non-informatif.
3. Normalisasi spasi.
4. Terapkan kamus normalisasi kata tidak baku jika digunakan.
5. Terapkan stopword removal secara selektif jika dipakai.
6. Simpan teks hasil ke kolom `content_clean`.
7. Buang atau tandai baris yang kosong setelah preprocessing.

## Checklist Validasi

- [x] Teks bersih tetap mempertahankan makna utama.
- [x] Tidak terlalu banyak ulasan menjadi string kosong.
- [x] Sampel hasil cleaning sudah dicek manual.
- [x] File output dapat dibaca ulang tanpa error encoding.

## Gate Pindah Fase

- [x] `preprocessed_reviews.csv` tersimpan di lokasi final.
- [x] Kolom `content_clean` konsisten dan tidak null.
- [x] Aturan preprocessing sudah dibekukan untuk eksperimen awal.

## Catatan Eksekusi 2026-03-17

- Script preprocessing: `src/preprocessing/preprocess_reviews.py`
- Command: `python -m src.preprocessing.preprocess_reviews`
- Input: `data/raw/raw_reviews.csv`
- Output utama: `data/processed/preprocessed_reviews.csv`
- Output ringkasan: `logs/preprocessing_summary.json` dan `logs/preprocessing_report.md`
- Aturan cleaning yang dibekukan untuk eksperimen awal: case folding, hapus URL/angka/tanda baca/emoji, normalisasi spasi, normalisasi slang dasar, selective stopword removal, dan buang baris kosong setelah cleaning.
- Kamus slang awal yang diterapkan mencakup bentuk umum seperti `gak/ga/gk/nggak -> tidak`, `yg -> yang`, `apk -> aplikasi`, `aja -> saja`, `tp -> tapi`, `sdh/udh -> sudah`, `dr -> dari`, dan `dgn -> dengan`.
- Selective stopword removal disetel konservatif dengan mempertahankan token negasi dan token pembawa sentimen penting seperti `tidak`, `bukan`, `belum`, `ada`, `bisa`, `baik`, `bagus`, `buruk`, `mudah`, `ribet`, dan `memuaskan`.
- Hasil eksekusi utama: 5000 baris input menjadi 4964 baris output; 36 baris (0,72%) dibuang karena kosong setelah cleaning.
- Validasi output: `content_clean` null = 0, `content_clean` blank = 0, file output berhasil dibaca ulang.
- Sampel cek manual menunjukkan frasa penting seperti `tidak ada solusi`, `tidak bisa`, dan `lebih baik` tetap terjaga setelah cleaning.

## Risiko Fase

- Cleaning terlalu agresif.
- Normalisasi slang tidak akurat.
- Banyak informasi hilang sebelum model dilatih.