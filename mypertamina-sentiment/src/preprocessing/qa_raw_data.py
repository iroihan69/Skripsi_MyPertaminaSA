from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    LOG_DIR,
    RAW_DATA_DIR,
    RAW_REVIEWS_FILENAME,
    SCRAPING_DEFAULT_COUNT,
    SCRAPING_END_DATE,
    SCRAPING_START_DATE,
)
from src.utils.io_utils import ensure_parent_directory
from src.utils.logging_utils import get_logger
from src.utils.validation_utils import validate_non_empty_text, validate_required_columns

REQUIRED_COLUMNS = [
    "reviewId",
    "content",
    "score",
    "at",
    "userName",
    "thumbsUpCount",
]

CRITICAL_COLUMNS = ["reviewId", "content", "score", "at"]
EXPECTED_SCORES = {1, 2, 3, 4, 5}
URL_PATTERN = re.compile(r"https?://|www\.", re.IGNORECASE)
NON_WORD_PATTERN = re.compile(r"[^\w\s]", re.UNICODE)


def _count_emoji_characters(text: str) -> int:
    total = 0
    for char in text:
        code_point = ord(char)
        if 0x1F300 <= code_point <= 0x1FAFF or 0x2600 <= code_point <= 0x27BF:
            total += 1
    return total


def _normalize_content(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _build_issue_flags(dataframe: pd.DataFrame) -> pd.DataFrame:
    duplicated_review_ids = dataframe.duplicated(subset=["reviewId"], keep="first")
    normalized_content = dataframe["content"].map(_normalize_content).str.lower()
    normalized_at = dataframe["at"].fillna("").astype(str).str.strip()
    duplicated_content_at = pd.DataFrame(
        {
            "content": normalized_content,
            "at": normalized_at,
        }
    ).duplicated(keep="first")

    parsed_score = pd.to_numeric(dataframe["score"], errors="coerce")
    parsed_at = pd.to_datetime(dataframe["at"], errors="coerce")

    flags: list[dict[str, Any]] = []
    for index, row in dataframe.iterrows():
        row_flags: list[str] = []
        content = _normalize_content(row.get("content"))
        review_id = _normalize_content(row.get("reviewId"))

        if not review_id:
            row_flags.append("missing_reviewId")
        if not validate_non_empty_text(content):
            row_flags.append("empty_content")
        if pd.isna(parsed_score.iloc[index]):
            row_flags.append("missing_or_invalid_score")
        elif int(parsed_score.iloc[index]) not in EXPECTED_SCORES:
            row_flags.append("unexpected_score")
        if pd.isna(parsed_at.iloc[index]):
            row_flags.append("invalid_at")
        if bool(duplicated_review_ids.iloc[index]):
            row_flags.append("duplicate_reviewId")
        if bool(duplicated_content_at.iloc[index]):
            row_flags.append("duplicate_content_at")

        if row_flags:
            flags.append(
                {
                    "reviewId": review_id,
                    "content": content,
                    "score": row.get("score"),
                    "at": row.get("at"),
                    "issueCount": len(row_flags),
                    "issueFlags": "|".join(row_flags),
                }
            )

    return pd.DataFrame(flags, columns=["reviewId", "content", "score", "at", "issueCount", "issueFlags"])


def build_quality_summary(dataframe: pd.DataFrame, minimum_rows: int) -> dict[str, Any]:
    missing_columns = validate_required_columns(dataframe.columns, REQUIRED_COLUMNS)
    parsed_score = pd.to_numeric(dataframe["score"], errors="coerce")
    parsed_at = pd.to_datetime(dataframe["at"], errors="coerce")
    normalized_content = dataframe["content"].map(_normalize_content)
    normalized_at = dataframe["at"].fillna("").astype(str).str.strip()

    duplicate_review_id_count = int(dataframe.duplicated(subset=["reviewId"], keep="first").sum())
    duplicate_content_at_count = int(
        pd.DataFrame(
            {
                "content": normalized_content.str.lower(),
                "at": normalized_at,
            }
        ).duplicated(keep="first").sum()
    )

    null_counts = {
        column: int(dataframe[column].isna().sum())
        for column in REQUIRED_COLUMNS
    }
    blank_content_count = int(normalized_content.eq("").sum())
    invalid_score_count = int(parsed_score.isna().sum() + (~parsed_score.dropna().astype(int).isin(EXPECTED_SCORES)).sum())
    invalid_at_count = int(parsed_at.isna().sum())

    start_boundary = pd.Timestamp(SCRAPING_START_DATE)
    end_boundary = pd.Timestamp(SCRAPING_END_DATE).replace(hour=23, minute=59, second=59)
    before_start_count = int((parsed_at.dropna() < start_boundary).sum())
    after_end_count = int((parsed_at.dropna() > end_boundary).sum())

    score_distribution = {
        str(score): {
            "count": int((parsed_score == score).sum()),
            "percentage": round(float((parsed_score == score).mean() * 100), 2),
        }
        for score in sorted(EXPECTED_SCORES)
    }

    url_count = int(normalized_content.map(lambda text: bool(URL_PATTERN.search(text))).sum())
    emoji_heavy_count = int(normalized_content.map(lambda text: _count_emoji_characters(text) >= 3).sum())
    symbol_heavy_count = int(normalized_content.map(lambda text: len(NON_WORD_PATTERN.findall(text)) >= 10).sum())

    flagged_rows = _build_issue_flags(dataframe)
    issue_counter = Counter()
    for raw_flags in flagged_rows["issueFlags"].tolist():
        for flag in raw_flags.split("|"):
            if flag:
                issue_counter[flag] += 1

    hard_blockers = []
    if missing_columns:
        hard_blockers.append("Kolom wajib belum lengkap")
    if len(dataframe) < minimum_rows:
        hard_blockers.append("Jumlah baris di bawah ambang minimum")
    if null_counts["score"] > 0 or invalid_score_count > 0:
        hard_blockers.append("Masih ada score null atau tidak valid")
    if invalid_at_count > 0:
        hard_blockers.append("Masih ada tanggal ulasan yang tidak valid")
    if blank_content_count > 0:
        hard_blockers.append("Masih ada content kosong")

    return {
        "checkedAt": datetime.now().isoformat(timespec="seconds"),
        "inputRows": int(len(dataframe)),
        "minimumRows": int(minimum_rows),
        "isMinimumRowsMet": bool(len(dataframe) >= minimum_rows),
        "missingRequiredColumns": missing_columns,
        "nullCounts": null_counts,
        "blankContentCount": blank_content_count,
        "duplicateReviewIdCount": duplicate_review_id_count,
        "duplicateContentAtCount": duplicate_content_at_count,
        "scoreDistribution": score_distribution,
        "invalidScoreCount": invalid_score_count,
        "invalidAtCount": invalid_at_count,
        "beforeStartDateCount": before_start_count,
        "afterEndDateCount": after_end_count,
        "dateRange": {
            "min": None if parsed_at.dropna().empty else parsed_at.min().isoformat(sep=" "),
            "max": None if parsed_at.dropna().empty else parsed_at.max().isoformat(sep=" "),
        },
        "noiseSummary": {
            "urlCount": url_count,
            "emojiHeavyCount": emoji_heavy_count,
            "symbolHeavyCount": symbol_heavy_count,
        },
        "flaggedRowCount": int(len(flagged_rows)),
        "issueCounts": dict(issue_counter),
        "hardBlockers": hard_blockers,
        "readyForPreprocessing": not hard_blockers,
        "decision": "Lanjut ke preprocessing" if not hard_blockers else "Perlu perbaikan sebelum preprocessing",
        "handlingRule": (
            "Pertahankan raw_reviews.csv apa adanya; gunakan raw_reviews_qa_flags.csv untuk menandai baris bermasalah sebelum dibuang atau dikecualikan di fase preprocessing."
        ),
    }


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_rows = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_row, separator_row, *body_rows])


def render_markdown_report(summary: dict[str, Any]) -> str:
    score_rows = [
        [
            score,
            str(values["count"]),
            f"{values['percentage']:.2f}%",
        ]
        for score, values in summary["scoreDistribution"].items()
    ]

    null_rows = [[column, str(count)] for column, count in summary["nullCounts"].items()]
    issue_rows = (
        [[flag, str(count)] for flag, count in summary["issueCounts"].items()]
        if summary["issueCounts"]
        else [["Tidak ada", "0"]]
    )
    blockers = summary["hardBlockers"] or ["Tidak ada blocker keras"]

    lines = [
        "# Laporan QA Raw Data",
        "",
        f"- Waktu cek: {summary['checkedAt']}",
        f"- Keputusan: {summary['decision']}",
        f"- Ready for preprocessing: {'Ya' if summary['readyForPreprocessing'] else 'Tidak'}",
        f"- Ambang minimum baris: {summary['minimumRows']}",
        f"- Total baris input: {summary['inputRows']}",
        "",
        "## Ringkasan Temuan",
        "",
        f"- Kolom wajib hilang: {', '.join(summary['missingRequiredColumns']) if summary['missingRequiredColumns'] else 'Tidak ada'}",
        f"- Null content/score/at: {summary['nullCounts']['content']}/{summary['nullCounts']['score']}/{summary['nullCounts']['at']}",
        f"- Content kosong setelah trim: {summary['blankContentCount']}",
        f"- Duplikasi reviewId: {summary['duplicateReviewIdCount']}",
        f"- Duplikasi kombinasi content-at: {summary['duplicateContentAtCount']}",
        f"- Tanggal invalid: {summary['invalidAtCount']}",
        f"- Noise URL/emoji berat/simbol berat: {summary['noiseSummary']['urlCount']}/{summary['noiseSummary']['emojiHeavyCount']}/{summary['noiseSummary']['symbolHeavyCount']}",
        "",
        "## Null pada Kolom Wajib",
        "",
        _markdown_table(["Kolom", "Jumlah Null"], null_rows),
        "",
        "## Distribusi Score",
        "",
        _markdown_table(["Score", "Jumlah", "Persentase"], score_rows),
        "",
        "## Ringkasan Isu yang Ditandai",
        "",
        _markdown_table(["Issue Flag", "Jumlah"], issue_rows),
        "",
        "## Validitas Tanggal",
        "",
        f"- Rentang tanggal aktual: {summary['dateRange']['min']} s.d. {summary['dateRange']['max']}",
        f"- Sebelum batas awal scraping ({SCRAPING_START_DATE}): {summary['beforeStartDateCount']}",
        f"- Sesudah batas akhir scraping ({SCRAPING_END_DATE}): {summary['afterEndDateCount']}",
        "",
        "## Aturan Penanganan Data Rusak",
        "",
        f"- {summary['handlingRule']}",
        "",
        "## Blocker Keras",
        "",
        *[f"- {item}" for item in blockers],
        "",
    ]
    return "\n".join(lines)


def save_outputs(summary: dict[str, Any], flagged_rows: pd.DataFrame, summary_path: Path, report_path: Path, flags_path: Path) -> None:
    ensure_parent_directory(summary_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    ensure_parent_directory(report_path)
    report_path.write_text(render_markdown_report(summary), encoding="utf-8")

    ensure_parent_directory(flags_path)
    flagged_rows.to_csv(flags_path, index=False, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quality check raw data ulasan MyPertamina")
    parser.add_argument(
        "--input",
        default=str(RAW_DATA_DIR / RAW_REVIEWS_FILENAME),
        help="Path file CSV raw reviews",
    )
    parser.add_argument(
        "--minimum-rows",
        type=int,
        default=SCRAPING_DEFAULT_COUNT,
        help="Ambang minimum jumlah baris agar lolos gate fase",
    )
    parser.add_argument(
        "--summary-output",
        default=str(LOG_DIR / "raw_data_qa_summary.json"),
        help="Path output JSON ringkasan QA",
    )
    parser.add_argument(
        "--report-output",
        default=str(LOG_DIR / "raw_data_qa_report.md"),
        help="Path output laporan markdown QA",
    )
    parser.add_argument(
        "--flags-output",
        default=str(RAW_DATA_DIR / "raw_reviews_qa_flags.csv"),
        help="Path output CSV flag issue per baris",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = get_logger("raw_data_qa", log_filename="raw_data_qa.log")

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"File raw data tidak ditemukan: {input_path}")

    dataframe = pd.read_csv(input_path)
    missing_columns = validate_required_columns(dataframe.columns, REQUIRED_COLUMNS)
    if missing_columns:
        raise ValueError(f"Kolom wajib hilang: {', '.join(missing_columns)}")

    summary = build_quality_summary(dataframe, minimum_rows=args.minimum_rows)
    flagged_rows = _build_issue_flags(dataframe)
    save_outputs(
        summary=summary,
        flagged_rows=flagged_rows,
        summary_path=Path(args.summary_output),
        report_path=Path(args.report_output),
        flags_path=Path(args.flags_output),
    )

    logger.info("QA raw data selesai")
    logger.info("Decision: %s", summary["decision"])
    logger.info("Flagged rows: %s", summary["flaggedRowCount"])
    logger.info("Report: %s", args.report_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())