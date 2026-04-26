from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    LOG_DIR,
    PREPROCESSED_REVIEWS_FILENAME,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    TEST_DATA_FILENAME,
    TRAIN_DATA_FILENAME,
    ensure_base_directories,
)
from src.utils.io_utils import ensure_parent_directory
from src.utils.logging_utils import get_logger
from src.utils.validation_utils import validate_required_columns

REQUIRED_COLUMNS = ["reviewId", "score", "content_clean"]
RATING_TO_LABEL_3CLASS = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}
LABEL_NAMES_3CLASS = {0: "Negatif", 1: "Netral", 2: "Positif"}
RATING_TO_LABEL_2CLASS = {1: 0, 2: 0, 3: 0, 4: 1, 5: 1}
LABEL_NAMES_2CLASS = {0: "Negatif", 1: "Positif"}


def get_label_scheme_config(label_scheme: str) -> tuple[dict[int, int], dict[int, str]]:
    if label_scheme == "2class":
        return RATING_TO_LABEL_2CLASS, LABEL_NAMES_2CLASS
    return RATING_TO_LABEL_3CLASS, LABEL_NAMES_3CLASS


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_rows = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_row, separator_row, *body_rows])


def _build_distribution(dataframe: pd.DataFrame, label_names: dict[int, str]) -> dict[str, dict[str, int]]:
    counts = Counter(int(label) for label in dataframe["label"].tolist())
    return {
        str(label): {
            "labelName": label_names[label],
            "count": int(counts.get(label, 0)),
        }
        for label in sorted(label_names)
    }


def _distribution_rows(distribution: dict[str, dict[str, int]], label_names: dict[int, str]) -> list[list[str]]:
    total = sum(item["count"] for item in distribution.values())
    rows: list[list[str]] = []
    for label in [str(item) for item in sorted(label_names)]:
        count = distribution[label]["count"]
        percentage = (count / total) * 100 if total else 0.0
        rows.append([label, distribution[label]["labelName"], str(count), f"{percentage:.2f}%"])
    return rows


def map_labels(
    dataframe: pd.DataFrame,
    rating_to_label: dict[int, int],
    label_names: dict[int, str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    work_df = dataframe.copy()
    work_df["score"] = pd.to_numeric(work_df["score"], errors="coerce")

    invalid_score_rows = work_df[work_df["score"].isna() | ~work_df["score"].isin(rating_to_label.keys())]
    if not invalid_score_rows.empty:
        invalid_values = sorted({str(value) for value in invalid_score_rows["score"].dropna().tolist()})
        raise ValueError(
            "Ditemukan score tidak valid untuk mapping label. "
            f"Jumlah baris: {len(invalid_score_rows)}, nilai unik: {invalid_values or ['NaN']}"
        )

    work_df["score"] = work_df["score"].astype(int)
    work_df["label"] = work_df["score"].map(rating_to_label).astype(int)
    work_df["label_name"] = work_df["label"].map(label_names)

    if work_df["label"].isna().any():
        raise ValueError("Terdapat baris tanpa label setelah proses mapping.")

    summary = {
        "inputRows": int(len(work_df)),
        "labelDistribution": _build_distribution(work_df, label_names),
    }
    return work_df, summary


def split_dataset(
    dataframe: pd.DataFrame,
    test_size: float,
    random_seed: int,
    holdout_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    if holdout_size > 0.0:
        remain_df, holdout_df = train_test_split(
            dataframe,
            test_size=holdout_size,
            random_state=random_seed,
            stratify=dataframe["label"],
            shuffle=True,
        )
        train_df, test_df = train_test_split(
            remain_df,
            test_size=test_size,
            random_state=random_seed,
            stratify=remain_df["label"],
            shuffle=True,
        )
        return (
            train_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
            holdout_df.reset_index(drop=True),
        )

    train_df, test_df = train_test_split(
        dataframe,
        test_size=test_size,
        random_state=random_seed,
        stratify=dataframe["label"],
        shuffle=True,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), None


def build_summary(
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    holdout_df: pd.DataFrame | None,
    input_path: Path,
    train_output_path: Path,
    test_output_path: Path,
    inference_output_path: Path | None,
    test_size: float,
    holdout_size: float,
    random_seed: int,
    label_scheme: str,
    label_names: dict[int, str],
) -> dict[str, Any]:
    overlap_train_test_count = int(
        len(set(train_df["reviewId"].astype(str)).intersection(set(test_df["reviewId"].astype(str))))
    )

    overlap_train_holdout_count = 0
    overlap_test_holdout_count = 0
    if holdout_df is not None:
        overlap_train_holdout_count = int(
            len(set(train_df["reviewId"].astype(str)).intersection(set(holdout_df["reviewId"].astype(str))))
        )
        overlap_test_holdout_count = int(
            len(set(test_df["reviewId"].astype(str)).intersection(set(holdout_df["reviewId"].astype(str))))
        )

    has_obvious_leakage = any(
        [
            overlap_train_test_count > 0,
            overlap_train_holdout_count > 0,
            overlap_test_holdout_count > 0,
        ]
    )

    return {
        "checkedAt": datetime.now().isoformat(timespec="seconds"),
        "inputPath": str(input_path),
        "trainOutputPath": str(train_output_path),
        "testOutputPath": str(test_output_path),
        "inferenceOutputPath": str(inference_output_path) if inference_output_path else None,
        "testSize": float(test_size),
        "holdoutSize": float(holdout_size),
        "randomSeed": int(random_seed),
        "labelScheme": label_scheme,
        "inputRows": int(len(full_df)),
        "trainRows": int(len(train_df)),
        "testRows": int(len(test_df)),
        "inferenceRows": int(len(holdout_df)) if holdout_df is not None else 0,
        "fullDistribution": _build_distribution(full_df, label_names),
        "trainDistribution": _build_distribution(train_df, label_names),
        "testDistribution": _build_distribution(test_df, label_names),
        "inferenceDistribution": _build_distribution(holdout_df, label_names) if holdout_df is not None else {},
        "labelColumnValid": bool(set(full_df["label"].unique()).issubset(set(label_names))),
        "reviewIdOverlapTrainTest": overlap_train_test_count,
        "reviewIdOverlapTrainInference": overlap_train_holdout_count,
        "reviewIdOverlapTestInference": overlap_test_holdout_count,
        "hasObviousLeakage": has_obvious_leakage,
        "imbalanceDecision": (
            "Gunakan baseline tanpa rebalancing pada Fase 07; evaluasi class weighting saat fine-tuning jika metrik kelas minoritas rendah."
            if label_scheme == "3class"
            else "Skema 2-kelas dipakai (1-3 Negatif, 4-5 Positif) untuk mengurangi bias kelas minoritas Netral."
        ),
    }


def render_markdown_report(summary: dict[str, Any]) -> str:
    label_names = {
        int(label_id): info["labelName"]
        for label_id, info in summary["fullDistribution"].items()
    }
    label_list_text = ", ".join(str(label) for label in sorted(label_names))

    lines = [
        "# Laporan Labeling dan Dataset Split",
        "",
        f"- Waktu proses: {summary['checkedAt']}",
        f"- Skema label: {summary['labelScheme']}",
        f"- Jumlah baris input: {summary['inputRows']}",
        f"- Jumlah baris train: {summary['trainRows']}",
        f"- Jumlah baris test: {summary['testRows']}",
        f"- Jumlah baris inference holdout: {summary['inferenceRows']}",
        f"- Proporsi test: {summary['testSize']}",
        f"- Proporsi inference holdout: {summary['holdoutSize']}",
        f"- Random seed: {summary['randomSeed']}",
        "",
        "## Distribusi Label Keseluruhan",
        "",
        _markdown_table(
            ["Label", "Nama", "Jumlah", "Persentase"],
            _distribution_rows(summary["fullDistribution"], label_names),
        ),
        "",
        "## Distribusi Label Train",
        "",
        _markdown_table(
            ["Label", "Nama", "Jumlah", "Persentase"],
            _distribution_rows(summary["trainDistribution"], label_names),
        ),
        "",
        "## Distribusi Label Test",
        "",
        _markdown_table(
            ["Label", "Nama", "Jumlah", "Persentase"],
            _distribution_rows(summary["testDistribution"], label_names),
        ),
        "",
        "## Distribusi Label Inference Holdout",
        "",
        _markdown_table(
            ["Label", "Nama", "Jumlah", "Persentase"],
            _distribution_rows(summary["inferenceDistribution"], label_names),
        )
        if summary["inferenceDistribution"]
        else "Inference holdout tidak diaktifkan.",
        "",
        "## Validasi",
        "",
        f"- Semua baris memiliki label numerik valid ({label_list_text}): {'Ya' if summary['labelColumnValid'] else 'Tidak'}",
        f"- Overlap reviewId train vs test: {summary['reviewIdOverlapTrainTest']}",
        f"- Overlap reviewId train vs inference holdout: {summary['reviewIdOverlapTrainInference']}",
        f"- Overlap reviewId test vs inference holdout: {summary['reviewIdOverlapTestInference']}",
        f"- Indikasi kebocoran data yang jelas: {'Ya' if summary['hasObviousLeakage'] else 'Tidak'}",
        f"- Keputusan awal imbalance: {summary['imbalanceDecision']}",
        "",
    ]
    return "\n".join(lines)


def save_outputs(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    holdout_df: pd.DataFrame | None,
    summary: dict[str, Any],
    train_output_path: Path,
    test_output_path: Path,
    inference_output_path: Path | None,
    summary_path: Path,
    report_path: Path,
) -> None:
    ensure_parent_directory(train_output_path)
    train_df.to_csv(train_output_path, index=False, encoding="utf-8")

    ensure_parent_directory(test_output_path)
    test_df.to_csv(test_output_path, index=False, encoding="utf-8")

    if holdout_df is not None and inference_output_path is not None:
        ensure_parent_directory(inference_output_path)
        holdout_df.to_csv(inference_output_path, index=False, encoding="utf-8")

    ensure_parent_directory(summary_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    ensure_parent_directory(report_path)
    report_path.write_text(render_markdown_report(summary), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Labeling sentimen dan split dataset MyPertamina")
    parser.add_argument(
        "--input",
        default=str(PROCESSED_DATA_DIR / PREPROCESSED_REVIEWS_FILENAME),
        help="Path file CSV hasil preprocessing",
    )
    parser.add_argument(
        "--train-output",
        default=str(PROCESSED_DATA_DIR / TRAIN_DATA_FILENAME),
        help="Path output file train_data.csv",
    )
    parser.add_argument(
        "--test-output",
        default=str(PROCESSED_DATA_DIR / TEST_DATA_FILENAME),
        help="Path output file test_data.csv",
    )
    parser.add_argument(
        "--summary-output",
        default=str(LOG_DIR / "label_split_summary.json"),
        help="Path output JSON ringkasan labeling/split",
    )
    parser.add_argument(
        "--report-output",
        default=str(LOG_DIR / "label_split_report.md"),
        help="Path output markdown laporan labeling/split",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proporsi data test (0-1)",
    )
    parser.add_argument(
        "--holdout-size",
        type=float,
        default=0.0,
        help="Proporsi inference holdout dari total data (0-1). Jika 0, holdout tidak dibuat.",
    )
    parser.add_argument(
        "--inference-output",
        default="",
        help="Path output file inference holdout CSV (wajib jika --holdout-size > 0)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=RANDOM_SEED,
        help="Seed random untuk split",
    )
    parser.add_argument(
        "--label-scheme",
        choices=["2class", "3class"],
        default="3class",
        help="Skema label: 3class (1-2 negatif, 3 netral, 4-5 positif) atau 2class (1-3 negatif, 4-5 positif)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = get_logger("label_and_split", log_filename="label_and_split.log")
    ensure_base_directories()

    if args.test_size <= 0.0 or args.test_size >= 1.0:
        logger.error("Nilai --test-size harus di antara 0 dan 1 (eksklusif).")
        return 1
    if args.holdout_size < 0.0 or args.holdout_size >= 1.0:
        logger.error("Nilai --holdout-size harus di antara 0 dan 1 (inklusif 0, eksklusif 1).")
        return 1
    if args.holdout_size > 0.0 and not args.inference_output.strip():
        logger.error("--inference-output wajib diisi jika --holdout-size > 0.")
        return 1
    if (1.0 - args.holdout_size) <= 0.0:
        logger.error("Sisa data setelah holdout harus lebih dari 0.")
        return 1

    input_path = Path(args.input)
    train_output_path = Path(args.train_output)
    test_output_path = Path(args.test_output)
    inference_output_path = Path(args.inference_output) if args.inference_output.strip() else None
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

    dataframe = dataframe.copy()
    dataframe["content_clean"] = dataframe["content_clean"].fillna("").astype(str)
    blank_mask = dataframe["content_clean"].str.strip().eq("")
    blank_content_clean_count = int(blank_mask.sum())
    if blank_content_clean_count > 0:
        logger.warning(
            "Ditemukan content_clean kosong sebanyak %s baris. Baris kosong akan dibuang sebelum split.",
            blank_content_clean_count,
        )
        dataframe = dataframe.loc[~blank_mask].reset_index(drop=True)

    rating_to_label, label_names = get_label_scheme_config(args.label_scheme)
    labeled_df, mapping_summary = map_labels(dataframe, rating_to_label, label_names)
    train_df, test_df, holdout_df = split_dataset(
        labeled_df,
        test_size=args.test_size,
        random_seed=args.random_seed,
        holdout_size=args.holdout_size,
    )

    summary = build_summary(
        full_df=labeled_df,
        train_df=train_df,
        test_df=test_df,
        holdout_df=holdout_df,
        input_path=input_path,
        train_output_path=train_output_path,
        test_output_path=test_output_path,
        inference_output_path=inference_output_path,
        test_size=args.test_size,
        holdout_size=args.holdout_size,
        random_seed=args.random_seed,
        label_scheme=args.label_scheme,
        label_names=label_names,
    )
    summary["mappingDistribution"] = mapping_summary["labelDistribution"]

    save_outputs(
        train_df,
        test_df,
        holdout_df,
        summary,
        train_output_path,
        test_output_path,
        inference_output_path,
        summary_path,
        report_path,
    )

    logger.info(
        "Labeling dan split selesai. Input=%s, train=%s, test=%s",
        len(labeled_df),
        len(train_df),
        len(test_df),
    )
    logger.info("Output train tersimpan di %s", train_output_path)
    logger.info("Output test tersimpan di %s", test_output_path)
    if holdout_df is not None and inference_output_path is not None:
        logger.info("Output inference holdout tersimpan di %s", inference_output_path)
    logger.info("Ringkasan labeling/split tersimpan di %s", summary_path)
    logger.info("Laporan labeling/split tersimpan di %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())