from __future__ import annotations

from pathlib import Path

# Root proyek mengacu ke folder mypertamina-sentiment/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Konstanta inti aplikasi
APP_ID = "com.dafturn.mypertamina"
LANG = "id"
COUNTRY = "id"
MODEL_NAME = "indobenchmark/indobert-base-p1"
RANDOM_SEED = 42
SCRAPING_DEFAULT_COUNT = 5000
SCRAPING_BATCH_SIZE = 200
SCRAPING_MAX_RETRIES = 3
SCRAPING_RETRY_BACKOFF_SECONDS = 2.0
SCRAPING_DELAY_SECONDS = 0.2
SCRAPING_START_DATE = "2022-01-01"
SCRAPING_END_DATE = "2025-12-31"

# Path utama proyek
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"

MODEL_OUTPUT_DIR = PROJECT_ROOT / "model_output"
LOG_DIR = PROJECT_ROOT / "logs"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Nama file output standar lintas fase
RAW_REVIEWS_FILENAME = "raw_reviews.csv"
PREPROCESSED_REVIEWS_FILENAME = "preprocessed_reviews.csv"
TRAIN_DATA_FILENAME = "train_data.csv"
TEST_DATA_FILENAME = "test_data.csv"
PREDICTIONS_FILENAME = "predictions.csv"


def ensure_base_directories() -> None:
    """Membuat direktori utama proyek jika belum tersedia."""
    for path in [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        PREDICTIONS_DIR,
        MODEL_OUTPUT_DIR,
        LOG_DIR,
        DASHBOARD_DIR,
        NOTEBOOKS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


STAGE_OUTPUT_FILES = {
    "scraping": RAW_DATA_DIR / RAW_REVIEWS_FILENAME,
    "preprocessing": PROCESSED_DATA_DIR / PREPROCESSED_REVIEWS_FILENAME,
    "split_train": PROCESSED_DATA_DIR / TRAIN_DATA_FILENAME,
    "split_test": PROCESSED_DATA_DIR / TEST_DATA_FILENAME,
    "prediction": PREDICTIONS_DIR / PREDICTIONS_FILENAME,
}