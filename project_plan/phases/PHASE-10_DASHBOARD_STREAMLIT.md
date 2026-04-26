# Fase 10 - Dashboard Streamlit

## Tujuan

Membangun dashboard visualisasi yang menampilkan hasil analisis sentimen secara jelas dan dapat dipakai untuk demo atau evaluasi.

## Dependensi

- Fase 09 selesai.

## Output Wajib

- Aplikasi `dashboard/app.py`.
- Komponen filter, metric cards, grafik distribusi, tren, dan tabel data.

## Checklist Masuk Fase

- [x] `predictions.csv` tersedia.
- [x] Struktur kolom untuk visualisasi sudah final.
- [x] Library Streamlit dan Plotly siap dipakai.

## Langkah Implementasi

1. Buat fungsi load data dengan caching.
2. Tambahkan filter tahun atau periode.
3. Tampilkan metric cards total ulasan dan jumlah per sentimen.
4. Tampilkan distribusi sentimen dengan pie chart atau bar chart.
5. Tampilkan tren sentimen per bulan atau tahun.
6. Tampilkan tabel data prediksi.
7. Jika tersedia, tampilkan confusion matrix model.

## Checklist Validasi

- [x] Dashboard dapat dijalankan lokal.
- [x] Semua komponen utama tampil.
- [x] Filter bekerja benar.
- [x] Format tanggal dan label mudah dibaca.

## Gate Pindah Fase

- [x] Dashboard stabil untuk demo lokal.
- [x] Tidak ada error load data saat file CSV valid.
- [x] Insight utama bisa dijelaskan dari tampilan dashboard.

## Risiko Fase

- Struktur CSV berubah setelah dashboard dibuat.
- Visualisasi tidak sinkron dengan label model.

## Catatan Eksekusi

- Tanggal eksekusi: 2026-03-24.
- Aplikasi dashboard diimplementasikan pada `dashboard/app.py`.
- Fitur utama yang tersedia:
	- Load data prediksi dengan caching (`@st.cache_data`) dari `data/predictions/predictions.csv`.
	- Filter sentimen, rentang tanggal, dan tahun dari kolom `at`.
	- Metric cards total ulasan, Negatif, Netral, Positif.
	- Visualisasi distribusi sentimen (pie chart dan bar chart).
	- Visualisasi tren sentimen per bulan (line chart).
	- Tabel data prediksi dengan format tanggal dan confidence yang mudah dibaca.
	- Confusion matrix model dari `logs/evaluation_summary.json` jika tersedia.
- Validasi teknis:
	- Sintaks script valid (`python -m py_compile dashboard/app.py`).
	- Startup Streamlit berhasil pada mode headless lokal (`python -m streamlit run dashboard/app.py --server.headless true --server.port 8510`).

## Catatan Eksekusi Revisi (2026-04-21)

- Dashboard revisi dibuat sebagai file baru tanpa mengubah `dashboard/app.py`.
- Nama file dashboard revisi: `dashboard/app_revisi.py`.
- Sumber data default diarahkan ke:
	- `data/predictions/predictions_revisi_2kelas_20260421.csv`
	- `logs/evaluation_summary_revisi_2kelas_20260421.json`
- Penambahan komponen visual:
	- Bar chart penyebaran sentimen Positif dan Negatif per tahun.
- Penambahan komponen evaluasi:
	- Kartu metrik Macro F1, Weighted F1, Recall Negatif, Recall Positif.
	- Confusion matrix dari summary evaluasi revisi.
	- Catatan limitation bahwa skema 3 kelas bersifat eksploratif.
- Validasi teknis revisi:
	- Sintaks script valid (`python -m py_compile dashboard/app_revisi.py`).
	- Startup Streamlit berhasil pada mode headless lokal (`python -m streamlit run dashboard/app_revisi.py --server.headless true --server.port 8511`).