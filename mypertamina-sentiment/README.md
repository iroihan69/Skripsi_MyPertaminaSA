# Sistem Analisis Sentimen Ulasan MyPertamina

Proyek skripsi/tugas akhir untuk klasifikasi sentimen ulasan pengguna aplikasi MyPertamina di Google Play Store menggunakan model **IndoBERT** dengan proses fine-tuning.

---

## Prasyarat

| Kebutuhan | Versi Minimum |
|---|---|
| Python | 3.13.x |
| pip | 23.x ke atas |

---

## Setup Lokal

### 1. Clone atau buka folder proyek

```bash
cd "d:\File Opieq\MypertaminaSA\mypertamina-sentiment"
```

### 2. Buat virtual environment

```bash
python -m venv .venv
```

### 3. Aktifkan virtual environment

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
.venv\Scripts\activate.bat
```

### 4. Install dependency

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Catatan PyTorch:** Jika menggunakan GPU NVIDIA, ikuti panduan instalasi PyTorch di https://pytorch.org/get-started/locally/ untuk mengganti baris `torch` di `requirements.txt` dengan versi CUDA yang sesuai.

---

## Struktur Proyek

```
mypertamina-sentiment/
├── data/
│   ├── raw/              # Data mentah hasil scraping
│   ├── processed/        # Data setelah preprocessing dan labeling
│   └── predictions/      # Hasil prediksi model
├── src/
│   ├── scraping/         # Modul pengambilan ulasan Google Play
│   ├── preprocessing/    # Pipeline pembersihan dan normalisasi teks
│   ├── modeling/         # Training, evaluasi, dan inferensi model
│   └── utils/            # Helper dan utilitas umum
├── model_output/         # Artefak model hasil fine-tuning (tidak di-commit)
├── dashboard/            # Aplikasi Streamlit
├── notebooks/            # Notebook eksplorasi
├── requirements.txt
└── README.md
```

---

## Menjalankan Pipeline (Ringkasan)

> Detail setiap langkah ada di folder `project_plan/phases/`.

| Langkah | Perintah |
|---|---|
| Scraping | `python -m src.scraping.scraper` |
| Preprocessing | `python src/preprocessing/preprocessor.py` |
| Training | `python src/modeling/train.py` |
| Evaluasi | `python src/modeling/evaluate.py` |
| Inferensi | `python src/modeling/predict.py` |
| Dashboard | `streamlit run dashboard/app.py` |

### Parameter Scraping (Fase 03)

Script scraping dijalankan dengan mode modul:

```bash
python -m src.scraping.scraper --count 5000 --start-date 2022-01-01 --end-date 2025-12-31 --app-id com.dafturn.mypertamina
```

Parameter utama:

| Parameter | Deskripsi | Default |
|---|---|---|
| `--app-id` | ID aplikasi Google Play | `com.dafturn.mypertamina` |
| `--lang` | Bahasa ulasan | `id` |
| `--country` | Negara sumber ulasan | `id` |
| `--count` | Target jumlah ulasan | `5000` |
| `--start-date` | Batas tanggal awal (`YYYY-MM-DD`) | `2022-01-01` |
| `--end-date` | Batas tanggal akhir (`YYYY-MM-DD`) | `2025-12-31` |
| `--batch-size` | Jumlah ulasan per request | `200` |
| `--sleep-seconds` | Delay antar request (detik) | `0.2` |
| `--max-retries` | Retry maksimum saat request gagal | `3` |
| `--retry-backoff-seconds` | Backoff dasar antar retry (detik) | `2.0` |
| `--output` | Lokasi file output CSV | `data/raw/raw_reviews.csv` |

---

## Referensi

- [IndoBERT — HuggingFace](https://huggingface.co/indobenchmark/indobert-base-p1)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [google-play-scraper](https://pypi.org/project/google-play-scraper/)
- [Streamlit Documentation](https://docs.streamlit.io)
