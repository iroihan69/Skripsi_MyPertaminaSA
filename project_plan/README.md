# Workspace Planning Implementasi

Folder ini dipakai sebagai ruang kerja implementasi yang terpisah dari TRD utama.

## Isi Folder

- `ROADMAP_IMPLEMENTASI.md` : gambaran fase dari awal sampai akhir.
- `IMPLEMENTATION_TRACKER.md` : tracker progres lintas fase.
- `BACKLOG_REVISI.md` : tempat mencatat saran revisi, asumsi, dan catatan di luar fase.
- `phases/` : file kerja detail per fase.

## Aturan Pakai

1. Kerjakan fase secara berurutan kecuali ada keputusan eksplisit untuk paralel.
2. Jangan pindah ke fase berikutnya sebelum bagian `Gate Pindah Fase` pada fase aktif sudah lengkap.
3. Jika menemukan perubahan kebutuhan, jangan ubah fase secara diam-diam.
   Catat dulu di `BACKLOG_REVISI.md`, lalu putuskan apakah perubahan itu masuk ke fase aktif atau fase berikutnya.
4. Update `IMPLEMENTATION_TRACKER.md` setiap kali satu output fase selesai.

## Urutan Kerja yang Disarankan

1. Baca `ROADMAP_IMPLEMENTASI.md`.
2. Mulai dari file fase aktif di folder `phases/`.
3. Tandai checklist implementasi dan validasi selama pengerjaan.
4. Jika gate fase sudah lolos, lanjut ke fase berikutnya.

## Catatan

Dokumen-dokumen ini ditulis untuk kebutuhan eksekusi bertahap dari TRD Sistem Analisis Sentimen MyPertamina berbasis IndoBERT.