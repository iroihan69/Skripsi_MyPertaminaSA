from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from google_play_scraper import Sort, reviews

from src.config import (
    APP_ID,
    COUNTRY,
    LANG,
    SCRAPING_BATCH_SIZE,
    SCRAPING_DEFAULT_COUNT,
    SCRAPING_DELAY_SECONDS,
    SCRAPING_END_DATE,
    SCRAPING_MAX_RETRIES,
    SCRAPING_RETRY_BACKOFF_SECONDS,
    SCRAPING_START_DATE,
)
from src.utils.io_utils import ensure_parent_directory
from src.utils.logging_utils import get_logger
from src.utils.naming_utils import get_output_path

REQUIRED_REVIEW_FIELDS = [
    "reviewId",
    "content",
    "score",
    "at",
    "userName",
    "thumbsUpCount",
]

OUTPUT_COLUMNS = REQUIRED_REVIEW_FIELDS + ["scrapedAt", "sourceAppId"]


def _parse_date_boundary(date_text: str | None, is_end: bool = False) -> datetime | None:
    if not date_text:
        return None

    parsed = datetime.strptime(date_text, "%Y-%m-%d")
    if is_end:
        return parsed.replace(hour=23, minute=59, second=59)
    return parsed


def _normalize_review(raw_review: dict[str, Any], scraped_at: datetime, app_id: str) -> dict[str, Any] | None:
    review_date = raw_review.get("at")
    if not isinstance(review_date, datetime):
        return None

    return {
        "reviewId": raw_review.get("reviewId", ""),
        "content": raw_review.get("content", ""),
        "score": raw_review.get("score"),
        "at": review_date.isoformat(sep=" "),
        "userName": raw_review.get("userName", ""),
        "thumbsUpCount": raw_review.get("thumbsUpCount", 0),
        "scrapedAt": scraped_at.isoformat(sep=" "),
        "sourceAppId": app_id,
    }


def scrape_reviews(
    app_id: str,
    lang: str = "id",
    country: str = "id",
    count: int = 5000,
    start_date: str | None = None,
    end_date: str | None = None,
    batch_size: int = 200,
    sleep_seconds: float = 0.2,
    max_retries: int = 3,
    retry_backoff_seconds: float = 2.0,
    logger_name: str = "scraper",
) -> pd.DataFrame:
    """Scrape ulasan Google Play sesuai kontrak TRD dan kembalikan DataFrame terfilter tanggal."""
    if count <= 0:
        raise ValueError("count harus lebih besar dari 0")
    if batch_size <= 0:
        raise ValueError("batch_size harus lebih besar dari 0")

    logger = get_logger(logger_name, log_filename="scraping.log")
    parsed_start = _parse_date_boundary(start_date, is_end=False)
    parsed_end = _parse_date_boundary(end_date, is_end=True)
    if parsed_start and parsed_end and parsed_start > parsed_end:
        raise ValueError("start_date tidak boleh lebih besar dari end_date")

    logger.info(
        "Mulai scraping app_id=%s lang=%s country=%s target=%s start_date=%s end_date=%s",
        app_id,
        lang,
        country,
        count,
        start_date,
        end_date,
    )

    scraped_at = datetime.now()
    continuation_token = None
    collected: list[dict[str, Any]] = []
    stop_pagination = False

    while len(collected) < count and not stop_pagination:
        current_batch_size = min(batch_size, count - len(collected))
        batch_result: list[dict[str, Any]] = []

        for attempt in range(1, max_retries + 1):
            try:
                batch_result, continuation_token = reviews(
                    app_id,
                    lang=lang,
                    country=country,
                    sort=Sort.NEWEST,
                    count=current_batch_size,
                    filter_score_with=None,
                    continuation_token=continuation_token,
                )
                break
            except Exception as error:  # pragma: no cover - dipicu oleh kondisi jaringan/live endpoint
                is_last_attempt = attempt == max_retries
                logger.warning(
                    "Gagal mengambil batch (attempt=%s/%s): %s",
                    attempt,
                    max_retries,
                    error,
                )
                if is_last_attempt:
                    raise RuntimeError("Scraping gagal setelah retry maksimum") from error
                time.sleep(retry_backoff_seconds * attempt)

        if not batch_result:
            logger.warning("Batch kosong diterima, scraping dihentikan")
            break

        for raw_review in batch_result:
            normalized = _normalize_review(raw_review, scraped_at=scraped_at, app_id=app_id)
            if normalized is None:
                continue

            review_date = datetime.fromisoformat(str(normalized["at"]))
            if parsed_end and review_date > parsed_end:
                continue

            if parsed_start and review_date < parsed_start:
                stop_pagination = True
                break

            collected.append(normalized)
            if len(collected) >= count:
                break

        logger.info("Progres scraping: %s/%s ulasan", len(collected), count)

        if continuation_token is None:
            logger.info("Continuation token habis, scraping dihentikan")
            break

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    dataframe = pd.DataFrame(collected, columns=OUTPUT_COLUMNS)
    if not dataframe.empty:
        dataframe.drop_duplicates(subset=["reviewId"], keep="first", inplace=True)
        dataframe.sort_values(by="at", ascending=False, inplace=True)
        dataframe.reset_index(drop=True, inplace=True)

    logger.info("Scraping selesai, total row valid=%s", len(dataframe))
    return dataframe


def save_scraped_reviews(dataframe: pd.DataFrame, output_path: Path | str) -> Path:
    output = ensure_parent_directory(output_path)
    dataframe.to_csv(output, index=False, encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scraping ulasan MyPertamina dari Google Play")
    parser.add_argument("--app-id", default=APP_ID, help="Application ID di Google Play")
    parser.add_argument("--lang", default=LANG, help="Bahasa scraping, default id")
    parser.add_argument("--country", default=COUNTRY, help="Negara scraping, default id")
    parser.add_argument("--count", type=int, default=SCRAPING_DEFAULT_COUNT, help="Target jumlah ulasan")
    parser.add_argument(
        "--start-date",
        default=SCRAPING_START_DATE,
        help="Filter tanggal awal (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        default=SCRAPING_END_DATE,
        help="Filter tanggal akhir (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=SCRAPING_BATCH_SIZE,
        help="Jumlah ulasan per request",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=SCRAPING_DELAY_SECONDS,
        help="Delay antar request dalam detik",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=SCRAPING_MAX_RETRIES,
        help="Retry maksimum jika request gagal",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=SCRAPING_RETRY_BACKOFF_SECONDS,
        help="Backoff dasar antar retry",
    )
    parser.add_argument(
        "--output",
        default=str(get_output_path("scraping")),
        help="Path output CSV",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = get_logger("scraper.cli", log_filename="scraping.log")

    dataframe = scrape_reviews(
        app_id=args.app_id,
        lang=args.lang,
        country=args.country,
        count=args.count,
        start_date=args.start_date,
        end_date=args.end_date,
        batch_size=args.batch_size,
        sleep_seconds=args.sleep_seconds,
        max_retries=args.max_retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
        logger_name="scraper",
    )

    output_path = save_scraped_reviews(dataframe, args.output)
    duplicate_count = len(dataframe) - dataframe["reviewId"].nunique() if not dataframe.empty else 0

    logger.info("CSV disimpan ke: %s", output_path)
    logger.info("Total baris final: %s", len(dataframe))
    logger.info("Duplikasi reviewId setelah dedup: %s", duplicate_count)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
