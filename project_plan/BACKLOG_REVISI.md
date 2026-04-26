# Backlog Revisi dan Catatan Non-Fase

Dokumen ini khusus untuk menyimpan saran revisi, asumsi, keputusan tertunda, atau catatan penting yang tidak sudah ditulis sebagai tugas langsung di file fase.

## Aturan Penulisan

1. Satu isu atau saran menggunakan satu blok entri.
2. Gunakan ID unik berurutan, misalnya `REV-001`, `REV-002`.
3. Jika catatan sudah berubah menjadi tugas implementasi yang jelas, pindahkan eksekusinya ke fase terkait.
4. Jangan pakai file ini untuk checklist pekerjaan rutin fase.

## Template Entri

```text
ID: REV-XXX
Tanggal: YYYY-MM-DD
Sumber: Dosen | Pembimbing | Penguji | Self-review | Uji coba | Lainnya
Kategori: Data | Model | Dashboard | Arsitektur | Evaluasi | Dokumentasi | Scope
Prioritas: Tinggi | Sedang | Rendah
Status: Open | Review | Deferred | Rejected | Done
Fase Terkait: 01-11 | N/A
Ringkasan:
Dampak Jika Diabaikan:
Saran Revisi:
Keputusan Sementara:
Tindak Lanjut:
PIC:
```

## Daftar Entri

### REV-001

ID: REV-001
Tanggal: 2026-03-16
Sumber: Self-review
Kategori: Arsitektur
Prioritas: Sedang
Status: Open
Fase Terkait: N/A
Ringkasan: Perlu diputuskan apakah scraping akan dijalankan penuh secara live saat demo atau memakai dataset hasil scraping yang sudah dibekukan.
Dampak Jika Diabaikan: Risiko demo gagal jika endpoint Google Play berubah atau koneksi tidak stabil.
Saran Revisi: Siapkan dua mode operasi, yaitu mode collect dan mode demo offline berbasis file CSV.
Keputusan Sementara: Tunda keputusan sampai fase scraping dan hardening.
Tindak Lanjut: Evaluasi setelah pipeline data berjalan stabil.
PIC: Pemilik proyek

---

### REV-007

ID: REV-007
Tanggal: 2026-03-16
Sumber: Uji coba
Kategori: Arsitektur
Prioritas: Tinggi
Status: Done
Fase Terkait: 03
Ringkasan: Nilai APP_ID pada TRD/Fase 03 (`com.pertamina.mobile`) tidak tersedia di endpoint Google Play saat pengujian (404), sehingga scraping menghasilkan 0 data.
Dampak Jika Diabaikan: Gate Fase 03 gagal karena `raw_reviews.csv` kosong dan pipeline data fase lanjut tidak dapat berjalan.
Saran Revisi: Gunakan APP_ID operasional `com.dafturn.mypertamina` untuk implementasi scraping, sambil mempertahankan catatan deviasi terhadap TRD.
Keputusan Sementara: Disetujui dan diterapkan pada konfigurasi proyek untuk Fase 03 agar scraping live menghasilkan data.
Tindak Lanjut: Konfirmasi ke dosen pembimbing apakah TRD perlu addendum resmi pada bagian sumber data aplikasi.
PIC: Pemilik proyek

---

### REV-006

ID: REV-006
Tanggal: 2026-03-16
Sumber: Self-review
Kategori: Dokumentasi
Prioritas: Rendah
Status: Done
Fase Terkait: N/A
Ringkasan: Update status pada `IMPLEMENTATION_TRACKER.md` menyentuh ringkasan lintas-fase (Fase 01 dan Fase 02), bukan artefak teknis inti Fase 02.
Dampak Jika Diabaikan: Status progres proyek bisa tidak sinkron dengan kondisi implementasi aktual.
Saran Revisi: Catat perubahan lintas-fase pada revision notes agar jejak keputusan tetap audit-able.
Keputusan Sementara: Perubahan tracker lintas-fase didokumentasikan melalui entri ini.
Tindak Lanjut: Untuk update status lintas-fase berikutnya, tetap tambahkan entri revision notes bila perubahan tidak murni output teknis fase aktif.
PIC: Pemilik proyek

---

### REV-002

ID: REV-002
Tanggal: 2026-03-16
Sumber: Self-review
Kategori: Arsitektur
Prioritas: Tinggi
Status: Open
Fase Terkait: 01, 07
Ringkasan: Versi minimum package di TRD (Appendix 9.2) tidak kompatibel dengan Python 3.13.3. Beberapa package perlu versi lebih baru.
Dampak Jika Diabaikan: `pip install` akan gagal atau menghasilkan conflict, terutama pada torch dan numpy.
Saran Revisi: Gunakan versi minimum yang sudah diperbarui di `requirements.txt`: torch>=2.4.0, numpy>=1.26.0, pandas>=2.0.0, transformers>=4.40.0. Versi-versi ini diterapkan pada Fase 01.
Keputusan Sementara: Versi baru sudah diterapkan di `requirements.txt`. TRD tidak diubah agar tetap jadi acuan referensi asli.
Tindak Lanjut: Verifikasi saat instalasi environment. Jika ada konflik lebih lanjut di Fase 07, perbarui catatan ini.
PIC: Pemilik proyek

---

### REV-003

ID: REV-003
Tanggal: 2026-03-16
Sumber: Self-review
Kategori: Arsitektur
Prioritas: Sedang
Status: Open
Fase Terkait: 01
Ringkasan: TRD tidak menyebutkan penggunaan virtual environment secara eksplisit. Pilihan jatuh ke `venv` (bukan `conda`) karena Python 3.13 lebih stabil dengan pip/venv.
Dampak Jika Diabaikan: Tidak ada dampak fungsional langsung, namun tanpa venv terdapat risiko konflik dependency global.
Saran Revisi: Standarkan penggunaan `.venv/` di root folder proyek. Sudah dicakup di `.gitignore` dan `README.md`.
Keputusan Sementara: Ditetapkan: venv dengan folder `.venv/` di dalam root proyek.
Tindak Lanjut: Tidak ada, keputusan sudah final.
PIC: Pemilik proyek

---

### REV-004

ID: REV-004
Tanggal: 2026-03-16
Sumber: Self-review
Kategori: Arsitektur
Prioritas: Rendah
Status: Open
Fase Terkait: 01, 03–09
Ringkasan: TRD tidak mencantumkan `__init__.py` di subfolder `src/`, namun file ini dibutuhkan agar modul bisa di-import antar-file dalam proyek.
Dampak Jika Diabaikan: Import relatif antar modul (`from src.utils import ...`) akan gagal saat runtime.
Saran Revisi: Tambahkan `__init__.py` kosong di `src/`, `src/scraping/`, `src/preprocessing/`, `src/modeling/`, `src/utils/`. Sudah dilakukan pada Fase 01.
Keputusan Sementara: Diterapkan pada Fase 01 sebagai bagian setup standar Python package.
Tindak Lanjut: Tidak ada tindak lanjut khusus.
PIC: Pemilik proyek
---

### REV-005

ID: REV-005
Tanggal: 2026-03-16
Sumber: Uji coba
Kategori: Arsitektur
Prioritas: Sedang
Status: Done
Fase Terkait: 01
Ringkasan: PowerShell Windows memblokir eksekusi skrip .ps1 (termasuk .venv\Scripts\Activate.ps1) karena execution policy default Restricted.
Dampak Jika Diabaikan: Virtual environment tidak bisa diaktifkan via PowerShell, proses setup terhenti.
Saran Revisi: Jalankan satu kali Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser. Hanya mengubah kebijakan untuk user saat ini, tidak system-wide. Aman dan reversibel.
Keputusan Sementara: Sudah diterapkan. Perlu ditambahkan ke README.md sebagai langkah prasyarat setup Windows.
Tindak Lanjut: Update README.md dengan catatan execution policy di bagian Setup Lokal. Dijadwalkan saat finalisasi Fase 11.
PIC: Pemilik proyek
