# TECHNICAL REQUIREMENTS DOCUMENT
## Sistem Analisis Sentimen Ulasan Pengguna Aplikasi MyPertamina Menggunakan Model IndoBERT

---

| Properti | Keterangan |
|---|---|
| **Dokumen** | Technical Requirements Document (TRD) |
| **Versi** | 1.0.0 |
| **Status** | Draft Final |
| **Tanggal** | Maret 2025 |
| **Konteks** | Skripsi / Tugas Akhir |

---

## Daftar Isi

1. [Pendahuluan](#1-pendahuluan)
2. [Arsitektur Sistem](#2-arsitektur-sistem)
3. [Arsitektur Data Pipeline (ETL)](#3-arsitektur-data-pipeline-etl)
4. [Spesifikasi Teknis Model IndoBERT](#4-spesifikasi-teknis-model-indobert)
5. [Spesifikasi Evaluasi Model](#5-spesifikasi-evaluasi-model)
6. [Spesifikasi API dan Integrasi Sistem](#6-spesifikasi-api-dan-integrasi-sistem)
7. [Spesifikasi Dashboard Visualisasi](#7-spesifikasi-dashboard-visualisasi)
8. [Risiko Teknis dan Mitigasi](#8-risiko-teknis-dan-mitigasi)
9. [Appendix](#9-appendix)

---

## 1. PENDAHULUAN

### 1.1 Tujuan Dokumen

Dokumen Technical Requirements Document (TRD) ini menjabarkan spesifikasi teknis lengkap dari sistem analisis sentimen ulasan pengguna aplikasi MyPertamina yang dikumpulkan dari Google Play Store. Dokumen ini mencakup arsitektur sistem, spesifikasi komponen, pipeline data, detail model machine learning, serta kebutuhan teknis yang diperlukan untuk implementasi sistem secara keseluruhan.

Dokumen ini ditujukan sebagai acuan teknis dalam penulisan skripsi/tugas akhir dan sebagai panduan implementasi sistem.

### 1.2 Ruang Lingkup Sistem

Sistem yang dikembangkan mencakup komponen-komponen berikut:

- Modul pengumpulan data ulasan dari Google Play Store secara otomatis (web scraping)
- Pipeline preprocessing teks berbahasa Indonesia
- Modul pelabelan sentimen berbasis rating
- Model klasifikasi sentimen menggunakan IndoBERT dengan proses fine-tuning
- Modul evaluasi performa model menggunakan metrik standar
- Dashboard visualisasi berbasis Streamlit

### 1.3 Definisi dan Akronim

| Istilah / Akronim | Definisi |
|---|---|
| **BERT** | Bidirectional Encoder Representations from Transformers – arsitektur model bahasa berbasis transformer |
| **IndoBERT** | Versi BERT yang telah di-pre-train pada korpus teks bahasa Indonesia |
| **Fine-tuning** | Proses pelatihan lanjutan model pre-trained pada dataset spesifik untuk task tertentu |
| **NLP** | Natural Language Processing – cabang AI yang menangani pemrosesan bahasa alami |
| **ETL** | Extract, Transform, Load – pipeline pemrosesan data dari sumber hingga penyimpanan |
| **HuggingFace** | Platform dan library open-source untuk model NLP berbasis transformer |
| **Streamlit** | Framework Python untuk membangun aplikasi web data science secara cepat |
| **TRD** | Technical Requirements Document |

---

## 2. ARSITEKTUR SISTEM

### 2.1 Gambaran Umum Arsitektur

Sistem analisis sentimen ini mengadopsi pendekatan pipeline machine learning yang bersifat sequential dan modular. Setiap komponen dirancang untuk dapat dijalankan secara independen maupun terintegrasi dalam satu alur pemrosesan end-to-end.

| # | Komponen | Fungsi Utama | Teknologi |
|---|---|---|---|
| 1 | Data Source | Sumber data ulasan Google Play Store | Google Play Store API |
| 2 | Data Collection | Web scraping otomatis ulasan pengguna | Python, google-play-scraper |
| 3 | Data Storage | Penyimpanan dataset terstruktur | CSV, Pandas DataFrame |
| 4 | Data Preprocessing | Pembersihan dan normalisasi teks | Python, NLTK, Sastrawi |
| 5 | Sentiment Labeling | Pelabelan sentimen berbasis rating | Python, Pandas |
| 6 | Model Training | Fine-tuning IndoBERT untuk klasifikasi | HuggingFace Transformers, PyTorch |
| 7 | Model Evaluation | Evaluasi performa model klasifikasi | Scikit-learn, Seaborn |
| 8 | Visualization | Dashboard analisis dan tren sentimen | Streamlit, Plotly, Matplotlib |

### 2.2 Alur Data (Data Flow)

Alur pemrosesan data dalam sistem berjalan secara linear dari sumber data menuju visualisasi akhir:

```
[ Google Play Store ]
        ↓
[ Web Scraping ]
        ↓
[ Raw Dataset (.csv) ]
        ↓
[ Preprocessing ]
        ↓
[ Labeled Dataset ]
        ↓
[ Train / Test Split (80:20) ]
        ↓
[ IndoBERT Fine-tuning ]
        ↓
[ Prediksi Sentimen ]
        ↓
[ Evaluasi Model ]
        ↓
[ Trend Analysis ]
        ↓
[ Streamlit Dashboard ]
```

### 2.3 Spesifikasi Lingkungan Teknis

Sistem dapat dijalankan pada lingkungan lokal maupun cloud. Berikut adalah spesifikasi minimum yang direkomendasikan:

| Komponen | Minimum | Rekomendasi |
|---|---|---|
| **CPU** | 4 Core | 8 Core (Intel i7 / AMD Ryzen 7) |
| **RAM** | 8 GB | 16 GB |
| **GPU (opsional)** | Tidak wajib (CPU mode) | NVIDIA GPU dengan CUDA support (≥ 6 GB VRAM) |
| **Storage** | 10 GB | 20 GB (SSD) |
| **Python** | 3.8+ | 3.10+ |
| **OS** | Windows 10 / Ubuntu 20.04 | Ubuntu 22.04 / macOS Monterey |

---

## 3. ARSITEKTUR DATA PIPELINE (ETL)

### 3.1 Extract – Pengumpulan Data

#### 3.1.1 Sumber Data

Data diambil dari halaman ulasan aplikasi MyPertamina (`com.pertamina.mobile`) pada Google Play Store. Data mencakup periode 2022–2025.

#### 3.1.2 Metode Scraping

Pengumpulan data menggunakan library `google-play-scraper` yang berkomunikasi langsung dengan endpoint Google Play Store tanpa memerlukan autentikasi API key.

```python
# Instalasi dependensi
pip install google-play-scraper pandas

# Implementasi scraping
from google_play_scraper import reviews, Sort
import pandas as pd

result, continuation_token = reviews(
    'com.pertamina.mobile',
    lang='id',
    country='id',
    sort=Sort.NEWEST,
    count=5000,
    filter_score_with=None
)

df = pd.DataFrame(result)
df.to_csv('raw_reviews.csv', index=False)
```

#### 3.1.3 Atribut Dataset Hasil Scraping

| Nama Kolom | Tipe Data | Deskripsi | Contoh Nilai |
|---|---|---|---|
| `reviewId` | string | ID unik ulasan | `gp:A0B1C2...` |
| `content` | string | Teks ulasan pengguna | `Aplikasi sangat membantu` |
| `score` | integer (1–5) | Rating bintang pengguna | `4` |
| `at` | datetime | Tanggal ulasan diposting | `2024-03-15 08:00:00` |
| `userName` | string | Nama pengguna (anonim) | `User123` |
| `thumbsUpCount` | integer | Jumlah like pada ulasan | `12` |

---

### 3.2 Transform – Preprocessing dan Labeling

#### 3.2.1 Pipeline Preprocessing

Preprocessing teks dilakukan secara berurutan melalui tahapan berikut:

| # | Tahap | Deskripsi | Contoh |
|---|---|---|---|
| 1 | **Case Folding** | Konversi seluruh teks ke huruf kecil | `"MyPertamina Bagus"` → `"mypertamina bagus"` |
| 2 | **Cleaning** | Hapus tanda baca, angka, URL, emoji, dan simbol | `"bagus!!! 👍"` → `"bagus"` |
| 3 | **Normalisasi** | Koreksi kata tidak baku / singkatan informal | `"gk bisa"` → `"tidak bisa"` |
| 4 | **Stopword Removal** | Hapus kata umum (opsional, selektif) | `"dan"`, `"yang"`, `"di"`, `"atau"` |
| 5 | **Tokenisasi** | Tokenisasi via AutoTokenizer IndoBERT | `"sangat membantu"` → `[token IDs]` |

```python
import re
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('indobenchmark/indobert-base-p1')

def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)    # hapus URL
    text = re.sub(r'[^a-zA-Z\s]', '', text)       # hapus non-alfabet
    text = re.sub(r'\s+', ' ', text).strip()       # normalisasi spasi
    return text

def tokenize(text, max_length=128):
    return tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
```

#### 3.2.2 Sentiment Labeling

Pelabelan sentimen menggunakan pendekatan rating-based labeling berdasarkan skor ulasan pengguna:

| Rating | Label Sentimen | Nilai Numerik |
|---|---|---|
| 1 – 2 bintang | Negatif | `0` |
| 3 bintang | Netral | `1` |
| 4 – 5 bintang | Positif | `2` |

#### 3.2.3 Dataset Splitting

Pembagian dataset menggunakan metode stratified train-test split untuk menjaga distribusi kelas yang proporsional:

```python
from sklearn.model_selection import train_test_split

X = df['content_clean']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y          # stratified sampling
)

# Distribusi: 80% training | 20% testing
```

---

### 3.3 Load – Penyimpanan Data

Data hasil preprocessing disimpan dalam format CSV dan dapat dimuat ulang sebagai Pandas DataFrame untuk keperluan pelatihan model:

| File | Format | Deskripsi |
|---|---|---|
| `raw_reviews.csv` | CSV | Data mentah hasil scraping |
| `preprocessed_reviews.csv` | CSV | Data setelah preprocessing dan labeling |
| `train_data.csv` | CSV | 80% data untuk pelatihan model |
| `test_data.csv` | CSV | 20% data untuk evaluasi model |
| `predictions.csv` | CSV | Output prediksi sentimen beserta tanggal |

---

## 4. SPESIFIKASI TEKNIS MODEL INDOBERT

### 4.1 Deskripsi Model

IndoBERT adalah model bahasa berbasis arsitektur BERT (Bidirectional Encoder Representations from Transformers) yang dikembangkan oleh tim indobenchmark dan telah di-pre-train pada korpus teks bahasa Indonesia berukuran besar. Model ini tersedia secara publik melalui platform HuggingFace.

| Properti Model | Nilai / Keterangan |
|---|---|
| **Model ID (Base)** | `indobenchmark/indobert-base-p1` |
| **Model ID (Large)** | `indobenchmark/indobert-large-p1` |
| **Arsitektur** | BERT-base (12 layers, 12 attention heads, 768 hidden size) |
| **Total Parameter** | ~124 juta parameter (base) |
| **Vocab Size** | 50.000 token (WordPiece) |
| **Max Sequence Length** | 512 token |
| **Bahasa Pre-training** | Bahasa Indonesia |
| **Task yang Disupport** | Klasifikasi teks, NER, QA, dan lainnya |
| **Source** | [huggingface.co/indobenchmark/indobert-base-p1](https://huggingface.co/indobenchmark/indobert-base-p1) |

### 4.2 Arsitektur Fine-tuning untuk Klasifikasi Sentimen

Untuk task klasifikasi sentimen 3 kelas (Negatif, Netral, Positif), model IndoBERT dikombinasikan dengan classification head sebagai berikut:

```
[ Input Token IDs ]
        ↓
[ IndoBERT Encoder (12 Transformer Layers) ]
        ↓
[ [CLS] Token Representation (768 dim) ]
        ↓
[ Dropout Layer (p=0.1) ]
        ↓
[ Linear Layer (768 → 3) ]
        ↓
[ Softmax Activation ]
        ↓
[ Output: { Negatif=0, Netral=1, Positif=2 } ]
```

```python
from transformers import AutoModelForSequenceClassification
import torch

MODEL_NAME = 'indobenchmark/indobert-base-p1'
NUM_LABELS = 3  # Negatif=0, Netral=1, Positif=2

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS
)
```

### 4.3 Hyperparameter Fine-tuning

Berikut adalah konfigurasi hyperparameter yang digunakan dalam proses fine-tuning model IndoBERT:

| Hyperparameter | Nilai Default | Keterangan |
|---|---|---|
| **Learning Rate** | `2e-5` | Nilai umum untuk fine-tuning BERT; gunakan `1e-5` atau `3e-5` jika perlu |
| **Batch Size (Training)** | `16` | Sesuaikan dengan kapasitas VRAM GPU; gunakan `8` jika memori terbatas |
| **Batch Size (Evaluation)** | `32` | Dapat lebih besar dari training batch size |
| **Epochs** | `3 – 5` | Umumnya 3 epoch cukup; pantau validation loss |
| **Max Sequence Length** | `128` | Ulasan pendek; 128 token biasanya sudah mencukupi |
| **Optimizer** | `AdamW` | Versi Adam dengan weight decay untuk regularisasi |
| **Weight Decay** | `0.01` | Regularisasi untuk mencegah overfitting |
| **Warmup Steps** | `500` | Linear warmup scheduler untuk stabilisasi awal training |
| **Dropout Rate** | `0.1` | Dropout pada classification head |
| **Scheduler** | `Linear decay` | Learning rate dikurangi secara linear setelah warmup |
| **Seed** | `42` | Random seed untuk reproducibility |

### 4.4 Implementasi Training Loop

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    learning_rate=2e-5,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    logging_dir='./logs',
    seed=42,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model('./model_output')
```

---

## 5. SPESIFIKASI EVALUASI MODEL

### 5.1 Metrik Evaluasi

Evaluasi performa model dilakukan menggunakan metrik standar klasifikasi yang dihitung dari confusion matrix:

| Metrik | Formula | Keterangan |
|---|---|---|
| **Accuracy** | `(TP + TN) / (TP + TN + FP + FN)` | Proporsi prediksi benar dari seluruh data |
| **Precision** | `TP / (TP + FP)` | Akurasi prediksi positif; relevan saat false positive mahal |
| **Recall** | `TP / (TP + FN)` | Kemampuan menemukan semua instance positif |
| **F1-Score** | `2 × (Precision × Recall) / (Precision + Recall)` | Harmonic mean Precision dan Recall; digunakan sebagai metrik utama |

### 5.2 Implementasi Metrik

```python
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    report = classification_report(
        labels, predictions,
        target_names=['Negatif', 'Netral', 'Positif'],
        output_dict=True
    )
    return {
        'accuracy': report['accuracy'],
        'f1': report['weighted avg']['f1-score'],
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
    }
```

### 5.3 Target Performa Model

| Metrik | Target Minimum | Target Ideal |
|---|---|---|
| **Accuracy** | ≥ 80% | ≥ 90% |
| **Precision (weighted)** | ≥ 78% | ≥ 88% |
| **Recall (weighted)** | ≥ 78% | ≥ 88% |
| **F1-Score (weighted)** | ≥ 78% | ≥ 88% |

---

## 6. SPESIFIKASI API DAN INTEGRASI SISTEM

### 6.1 Antarmuka Antar Komponen

Sistem tidak mengekspos REST API eksternal, namun memiliki antarmuka internal antar modul yang didefinisikan sebagai fungsi Python dengan kontrak input-output yang jelas.

#### 6.1.1 Modul Scraping → Storage

| Parameter | Spesifikasi |
|---|---|
| **Fungsi** | `scrape_reviews(app_id, lang, country, count, start_date, end_date)` |
| **Input** | `app_id: str`, `lang: str='id'`, `country: str='id'`, `count: int=5000` |
| **Output** | `pd.DataFrame` dengan kolom: `reviewId`, `content`, `score`, `at`, `userName`, `thumbsUpCount` |
| **Output File** | `raw_reviews.csv` |

#### 6.1.2 Modul Preprocessing → Labeled Dataset

| Parameter | Spesifikasi |
|---|---|
| **Fungsi** | `preprocess_dataset(df, text_col, score_col)` |
| **Input** | `df: pd.DataFrame`, `text_col: str='content'`, `score_col: str='score'` |
| **Output** | `pd.DataFrame` dengan kolom tambahan: `content_clean (str)`, `label (int: 0/1/2)` |
| **Output File** | `preprocessed_reviews.csv` |

#### 6.1.3 Modul Model → Prediksi Sentimen

| Parameter | Spesifikasi |
|---|---|
| **Fungsi** | `predict_sentiment(texts, model_path)` |
| **Input** | `texts: List[str]`, `model_path: str` (path ke direktori model) |
| **Output** | `List[dict]` berisi: `{text, predicted_label, confidence, label_name}` |
| **Output File** | `predictions.csv` dengan kolom: `content`, `predicted_label`, `label_name`, `confidence`, `at` |

### 6.2 Integrasi dengan Streamlit Dashboard

Streamlit berfungsi sebagai lapisan presentasi yang membaca file `predictions.csv` dan `preprocessed_reviews.csv` untuk menampilkan hasil analisis. Komunikasi antara model dan dashboard bersifat file-based (tidak real-time).

```python
# app.py – Streamlit Dashboard
import streamlit as st
import pandas as pd
import plotly.express as px

@st.cache_data
def load_data(path='predictions.csv'):
    df = pd.read_csv(path)
    df['at'] = pd.to_datetime(df['at'])
    df['year'] = df['at'].dt.year
    df['month'] = df['at'].dt.to_period('M')
    return df

df = load_data()

# Sidebar filter
year_filter = st.sidebar.multiselect('Pilih Tahun', options=df['year'].unique())
if year_filter:
    df = df[df['year'].isin(year_filter)]

# Statistik ringkasan
col1, col2, col3, col4 = st.columns(4)
col1.metric('Total Ulasan', len(df))
col2.metric('Positif', len(df[df['label_name'] == 'Positif']))
col3.metric('Negatif', len(df[df['label_name'] == 'Negatif']))
col4.metric('Netral', len(df[df['label_name'] == 'Netral']))
```

### 6.3 Dependency dan Library

| Library / Package | Versi Min. | Fungsi |
|---|---|---|
| `transformers` | 4.30+ | Loading dan fine-tuning model IndoBERT |
| `torch` (PyTorch) | 2.0+ | Backend deep learning untuk training |
| `google-play-scraper` | 1.2+ | Pengambilan ulasan dari Google Play Store |
| `pandas` | 1.5+ | Manipulasi dan penyimpanan dataset |
| `scikit-learn` | 1.2+ | Train-test split dan metrik evaluasi |
| `streamlit` | 1.25+ | Dashboard visualisasi interaktif |
| `plotly` | 5.0+ | Grafik interaktif pada dashboard |
| `matplotlib` / `seaborn` | 3.5+ / 0.12+ | Confusion matrix dan grafik evaluasi |
| `PySastrawi` | 1.0+ | Stemming bahasa Indonesia (opsional) |
| `nltk` | 3.8+ | Stopword removal bahasa Indonesia (opsional) |

---

## 7. SPESIFIKASI DASHBOARD VISUALISASI

### 7.1 Komponen Dashboard Streamlit

| Komponen | Tipe Visualisasi | Data yang Ditampilkan |
|---|---|---|
| **Statistik Dataset** | Metric Cards | Total ulasan, jumlah per kelas sentimen |
| **Distribusi Sentimen** | Pie Chart + Bar Chart | Proporsi Positif / Netral / Negatif |
| **Tren Sentimen** | Line Chart (per bulan/tahun) | Perubahan jumlah sentimen dari waktu ke waktu |
| **Perbandingan Periode** | Grouped Bar Chart | Sentimen per tahun (2022–2025) |
| **Confusion Matrix** | Heatmap | Hasil evaluasi model klasifikasi |
| **Tabel Data** | Interactive Table | Ulasan dengan prediksi dan confidence score |

### 7.2 Cara Menjalankan Dashboard

```bash
# Install dependencies
pip install -r requirements.txt

# Jalankan dashboard
streamlit run dashboard/app.py

# Dashboard akan terbuka di browser: http://localhost:8501
```

---

## 8. RISIKO TEKNIS DAN MITIGASI

| # | Risiko | Level | Strategi Mitigasi |
|---|---|---|---|
| 1 | Scraping diblokir oleh Google Play | **Tinggi** | Batasi request per menit; gunakan delay antar request; simpan hasil scraping secara lokal |
| 2 | Ketidakseimbangan kelas (class imbalance) | **Sedang** | Gunakan stratified sampling, class weight pada loss function, atau oversampling (SMOTE) |
| 3 | GPU tidak tersedia / memori tidak cukup | **Sedang** | Kurangi batch size; gunakan gradient accumulation; jalankan di CPU (lebih lambat) |
| 4 | Overfitting pada training set | **Sedang** | Gunakan early stopping; pantau validation loss; aktifkan dropout |
| 5 | Label noise dari rating-based labeling | **Rendah** | Verifikasi manual sampel ulasan rating 3 (netral); pertimbangkan anotasi manual jika perlu |

---

## 9. APPENDIX

### 9.1 Struktur Direktori Proyek

```
mypertamina-sentiment/
├── data/
│   ├── raw/
│   │   └── raw_reviews.csv
│   ├── processed/
│   │   ├── preprocessed_reviews.csv
│   │   ├── train_data.csv
│   │   └── test_data.csv
│   └── predictions/
│       └── predictions.csv
├── src/
│   ├── scraping/
│   │   └── scraper.py
│   ├── preprocessing/
│   │   └── preprocessor.py
│   ├── modeling/
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── evaluate.py
│   └── utils/
│       └── helpers.py
├── model_output/
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer/
├── dashboard/
│   └── app.py
├── notebooks/
│   └── exploratory_analysis.ipynb
├── requirements.txt
└── README.md
```

### 9.2 Contoh `requirements.txt`

```
transformers>=4.30.0
torch>=2.0.0
google-play-scraper>=1.2.0
pandas>=1.5.0
scikit-learn>=1.2.0
streamlit>=1.25.0
plotly>=5.0.0
matplotlib>=3.5.0
seaborn>=0.12.0
PySastrawi>=1.0.0
nltk>=3.8.0
numpy>=1.23.0
```

### 9.3 Referensi

- Koto, F., et al. (2020). *IndoNLU: Benchmark and Resources for Evaluating Indonesian Natural Language Understanding*. Proceedings of AACL-IJCNLP.
- Devlin, J., et al. (2018). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. arXiv:1810.04805.
- HuggingFace Transformers Documentation. https://huggingface.co/docs/transformers
- Streamlit Documentation. https://docs.streamlit.io
- google-play-scraper. https://pypi.org/project/google-play-scraper/

---

*TRD v1.0  |  Skripsi / Tugas Akhir  |  Maret 2025*
