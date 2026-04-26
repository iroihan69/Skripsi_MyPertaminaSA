from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from src.config import (
    LOG_DIR,
    PREPROCESSED_REVIEWS_FILENAME,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    RAW_REVIEWS_FILENAME,
    ensure_base_directories,
)
from src.utils.io_utils import ensure_parent_directory
from src.utils.logging_utils import get_logger
from src.utils.validation_utils import validate_required_columns

REQUIRED_COLUMNS = ["reviewId", "content", "score", "at"]
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MULTISPACE_PATTERN = re.compile(r"\s+")

DEFAULT_SLANG_MAP = {
    "aja": "saja",
    "aj": "saja",
    "apk": "aplikasi",
    "bgt": "banget",
    "bgtt": "banget",
    "dr": "dari",
    "dgn": "dengan",
    "gak": "tidak",
    "ga": "tidak",
    "gk": "tidak",
    "gx": "tidak",
    "jg": "juga",
    "jd": "jadi",
    "krn": "karena",
    "ngga": "tidak",
    "nggak": "tidak",
    "sdh": "sudah",
    "sm": "sama",
    "sy": "saya",
    "tdk": "tidak",
    "tp": "tapi",
    "udh": "sudah",
    "udhh": "sudah",
    "yg": "yang",
}

PRESERVED_TOKENS = {
    "ada",
    "aman",
    "banget",
    "bagus",
    "baik",
    "belum",
    "bisa",
    "bukan",
    "buruk",
    "cepat",
    "error",
    "gangguan",
    "jangan",
    "jelek",
    "kecewa",
    "kurang",
    "lambat",
    "lancar",
    "lebih",
    "mantap",
    "masih",
    "memuaskan",
    "mudah",
    "normal",
    "parah",
    "ramah",
    "ribet",
    "sangat",
    "sekali",
    "sulit",
    "tak",
    "tanpa",
    "tidak",
    "tolol",
}


def _normalize_raw_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _strip_noise_characters(text: str) -> str:
    text = URL_PATTERN.sub(" ", text.casefold())
    cleaned_characters: list[str] = []
    for char in text:
        if char.isalpha() or char.isspace():
            cleaned_characters.append(char)
        else:
            cleaned_characters.append(" ")
    return MULTISPACE_PATTERN.sub(" ", "".join(cleaned_characters)).strip()


def _build_stopword_set() -> set[str]:
    factory = StopWordRemoverFactory()
    stopwords = set(factory.get_stop_words())
    return stopwords - PRESERVED_TOKENS


def _normalize_tokens(tokens: list[str], slang_map: dict[str, str]) -> tuple[list[str], Counter[str]]:
    normalized_tokens: list[str] = []
    replacements: Counter[str] = Counter()

    for token in tokens:
        normalized = slang_map.get(token, token)
        if normalized != token:
            replacements[f"{token}->{normalized}"] += 1
        normalized_tokens.append(normalized)

    return normalized_tokens, replacements


def _remove_selective_stopwords(tokens: list[str], stopwords: set[str]) -> tuple[list[str], int]:
    filtered_tokens = [token for token in tokens if token not in stopwords]
    return filtered_tokens, len(tokens) - len(filtered_tokens)


def preprocess_dataframe(dataframe: pd.DataFrame, stopword_mode: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    stopwords = _build_stopword_set() if stopword_mode == "selective" else set()
    slang_replacement_counter: Counter[str] = Counter()
    dropped_indices: list[int] = []
    cleaned_texts: list[str] = []
    original_token_total = 0
    cleaned_token_total = 0
    removed_stopword_total = 0

    for index, raw_content in dataframe["content"].items():
        normalized_content = _normalize_raw_text(raw_content)
        normalized_for_tokenization = _strip_noise_characters(normalized_content)
        raw_tokens = normalized_for_tokenization.split() if normalized_for_tokenization else []
        original_token_total += len(raw_tokens)

        normalized_tokens, row_replacements = _normalize_tokens(raw_tokens, DEFAULT_SLANG_MAP)
        slang_replacement_counter.update(row_replacements)

        cleaned_tokens = normalized_tokens
        removed_stopwords = 0
        if stopword_mode == "selective":
            cleaned_tokens, removed_stopwords = _remove_selective_stopwords(normalized_tokens, stopwords)
            removed_stopword_total += removed_stopwords

        cleaned_text = " ".join(cleaned_tokens).strip()
        if not cleaned_text:
            dropped_indices.append(index)
            cleaned_texts.append("")
            continue

        cleaned_texts.append(cleaned_text)
        cleaned_token_total += len(cleaned_tokens)

    output_dataframe = dataframe.copy()
    output_dataframe["content_clean"] = cleaned_texts
    output_dataframe = output_dataframe.drop(index=dropped_indices).reset_index(drop=True)

    summary = {
        "checkedAt": datetime.now().isoformat(timespec="seconds"),
        "inputRows": int(len(dataframe)),
        "outputRows": int(len(output_dataframe)),
        "droppedEmptyRows": int(len(dropped_indices)),
        "dropRatePercentage": round(float((len(dropped_indices) / len(dataframe)) * 100), 2) if len(dataframe) else 0.0,
        "stopwordMode": stopword_mode,
        "stopwordRemovedTokenCount": int(removed_stopword_total),
        "originalTokenCount": int(original_token_total),
        "cleanedTokenCount": int(cleaned_token_total),
        "averageTokensBefore": round(float(original_token_total / len(dataframe)), 2) if len(dataframe) else 0.0,
        "averageTokensAfter": round(float(cleaned_token_total / len(output_dataframe)), 2) if len(output_dataframe) else 0.0,
        "contentCleanNullCount": int(output_dataframe["content_clean"].isna().sum()),
        "contentCleanBlankCount": int(output_dataframe["content_clean"].eq("").sum()),
        "slangReplacementCounts": dict(slang_replacement_counter.most_common()),
        "isOutputReadable": True,
        "preprocessingRules": {
            "caseFolding": True,
            "removeUrls": True,
            "removeDigits": True,
            "removePunctuation": True,
            "removeEmoji": True,
            "normalizeWhitespace": True,
            "slangNormalization": "default-basic",
            "stopwordRemoval": stopword_mode,
            "dropEmptyRows": True,
        },
    }

    return output_dataframe, summary


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_rows = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_row, separator_row, *body_rows])


def render_markdown_report(summary: dict[str, Any]) -> str:
    replacement_rows = (
        [[token_pair, str(count)] for token_pair, count in list(summary["slangReplacementCounts"].items())[:15]]
        if summary["slangReplacementCounts"]
        else [["Tidak ada", "0"]]
    )

    rules_rows = [[rule, "Ya" if value is True else str(value)] for rule, value in summary["preprocessingRules"].items()]

    lines = [
        "# Laporan Preprocessing Teks",
        "",
        f"- Waktu proses: {summary['checkedAt']}",
        f"- Total baris input: {summary['inputRows']}",
        f"- Total baris output: {summary['outputRows']}",
        f"- Baris dibuang karena kosong setelah cleaning: {summary['droppedEmptyRows']} ({summary['dropRatePercentage']:.2f}%)",
        f"- Mode stopword removal: {summary['stopwordMode']}",
        f"- Token stopword yang dihapus: {summary['stopwordRemovedTokenCount']}",
        "",
        "## Validasi Output",
        "",
        f"- `content_clean` null: {summary['contentCleanNullCount']}",
        f"- `content_clean` blank: {summary['contentCleanBlankCount']}",
        f"- Rata-rata token sebelum cleaning: {summary['averageTokensBefore']}",
        f"- Rata-rata token sesudah cleaning: {summary['averageTokensAfter']}",
        f"- File output dapat dibaca ulang: {'Ya' if summary['isOutputReadable'] else 'Tidak'}",
        "",
        "## Aturan Preprocessing",
        "",
        _markdown_table(["Aturan", "Nilai"], rules_rows),
        "",
        "## Ringkasan Normalisasi Slang",
        "",
        _markdown_table(["Penggantian", "Frekuensi"], replacement_rows),
        "",
    ]
    return "\n".join(lines)


def save_outputs(dataframe: pd.DataFrame, summary: dict[str, Any], output_path: Path, summary_path: Path, report_path: Path) -> None:
    ensure_parent_directory(output_path)
    dataframe.to_csv(output_path, index=False, encoding="utf-8")

    ensure_parent_directory(summary_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    ensure_parent_directory(report_path)
    report_path.write_text(render_markdown_report(summary), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocessing teks ulasan MyPertamina")
    parser.add_argument(
        "--input",
        default=str(RAW_DATA_DIR / RAW_REVIEWS_FILENAME),
        help="Path file CSV raw reviews",
    )
    parser.add_argument(
        "--output",
        default=str(PROCESSED_DATA_DIR / PREPROCESSED_REVIEWS_FILENAME),
        help="Path output file CSV hasil preprocessing",
    )
    parser.add_argument(
        "--summary-output",
        default=str(LOG_DIR / "preprocessing_summary.json"),
        help="Path output JSON ringkasan preprocessing",
    )
    parser.add_argument(
        "--report-output",
        default=str(LOG_DIR / "preprocessing_report.md"),
        help="Path output laporan markdown preprocessing",
    )
    parser.add_argument(
        "--stopword-mode",
        choices=["none", "selective"],
        default="selective",
        help="Mode penghapusan stopword",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = get_logger("preprocessing_reviews", log_filename="preprocessing_reviews.log")
    ensure_base_directories()

    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary_output)
    report_path = Path(args.report_output)

    if not input_path.exists():
        logger.error("File input tidak ditemukan: %s", input_path)
        return 1

    dataframe = pd.read_csv(input_path)
    missing_columns = validate_required_columns(dataframe.columns, REQUIRED_COLUMNS)
    if missing_columns:
        logger.error("Kolom wajib belum lengkap: %s", ", ".join(missing_columns))
        return 1

    processed_dataframe, summary = preprocess_dataframe(dataframe, stopword_mode=args.stopword_mode)

    if processed_dataframe.empty:
        logger.error("Semua baris menjadi kosong setelah preprocessing; hentikan output.")
        return 1

    save_outputs(processed_dataframe, summary, output_path, summary_path, report_path)

    reloaded = pd.read_csv(output_path)
    summary["isOutputReadable"] = True
    summary["reloadedRowCount"] = int(len(reloaded))
    summary["reloadedContentCleanNullCount"] = int(reloaded["content_clean"].isna().sum())
    save_outputs(processed_dataframe, summary, output_path, summary_path, report_path)

    logger.info("Preprocessing selesai. Input=%s, output=%s, dropped=%s", len(dataframe), len(processed_dataframe), summary["droppedEmptyRows"])
    logger.info("Ringkasan preprocessing tersimpan di %s", summary_path)
    logger.info("Laporan preprocessing tersimpan di %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())