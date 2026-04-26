from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import (
    LOG_DIR,
    MODEL_OUTPUT_DIR,
    PREDICTIONS_DIR,
    PREDICTIONS_FILENAME,
    PROCESSED_DATA_DIR,
    TEST_DATA_FILENAME,
    ensure_base_directories,
)
from src.utils.io_utils import ensure_parent_directory
from src.utils.logging_utils import get_logger

LABEL_NAMES_3CLASS = {0: "Negatif", 1: "Netral", 2: "Positif"}
LABEL_NAMES_2CLASS = {0: "Negatif", 1: "Positif"}


def get_label_names(label_scheme: str) -> dict[int, str]:
    if label_scheme == "2class":
        return LABEL_NAMES_2CLASS
    return LABEL_NAMES_3CLASS


@dataclass(slots=True)
class InferenceConfig:
    model_dir: str
    input_path: str
    output_path: str
    summary_output: str
    report_output: str
    text_column: str
    batch_size: int
    max_length: int
    label_scheme: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inferensi batch IndoBERT untuk Fase 09")
    parser.add_argument(
        "--model-dir",
        default=str(MODEL_OUTPUT_DIR / "baseline_indobert"),
        help="Direktori model hasil training",
    )
    parser.add_argument(
        "--input",
        default=str(PROCESSED_DATA_DIR / TEST_DATA_FILENAME),
        help="Path input CSV untuk inferensi batch",
    )
    parser.add_argument(
        "--output",
        default=str(PREDICTIONS_DIR / PREDICTIONS_FILENAME),
        help="Path output predictions.csv",
    )
    parser.add_argument(
        "--summary-output",
        default=str(LOG_DIR / "inference_summary.json"),
        help="Path output JSON ringkasan inferensi",
    )
    parser.add_argument(
        "--report-output",
        default=str(LOG_DIR / "inference_report.md"),
        help="Path output markdown laporan inferensi",
    )
    parser.add_argument(
        "--text-column",
        default="content_clean",
        help="Nama kolom teks yang dipakai untuk inferensi",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size inferensi")
    parser.add_argument("--max-length", type=int, default=128, help="Panjang token maksimum")
    parser.add_argument(
        "--label-scheme",
        choices=["2class", "3class"],
        default="3class",
        help="Skema label hasil prediksi",
    )
    return parser.parse_args()


def to_config(args: argparse.Namespace) -> InferenceConfig:
    return InferenceConfig(
        model_dir=str(Path(args.model_dir).resolve()),
        input_path=str(Path(args.input).resolve()),
        output_path=str(Path(args.output).resolve()),
        summary_output=str(Path(args.summary_output).resolve()),
        report_output=str(Path(args.report_output).resolve()),
        text_column=str(args.text_column),
        batch_size=max(1, int(args.batch_size)),
        max_length=max(8, int(args.max_length)),
        label_scheme=args.label_scheme,
    )


def load_input_dataframe(path: Path, text_column: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File input inferensi tidak ditemukan: {path}")

    dataframe = pd.read_csv(path)
    if text_column not in dataframe.columns:
        raise ValueError(f"Kolom teks '{text_column}' tidak ditemukan pada file: {path}")

    work_df = dataframe.copy()
    work_df[text_column] = work_df[text_column].fillna("").astype(str)

    blank_count = int(work_df[text_column].str.strip().eq("").sum())
    if blank_count > 0:
        raise ValueError(f"Ditemukan {blank_count} baris dengan teks kosong di kolom '{text_column}'")

    return work_df.reset_index(drop=True)


def predict_sentiment(
    texts: list[str],
    model_path: str | Path,
    *,
    batch_size: int = 32,
    max_length: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    pred_ids: list[np.ndarray] = []
    confidences: list[np.ndarray] = []
    probs_all: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            batch = tokenizer(
                batch_texts,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}

            outputs = model(**batch)
            probs = torch.softmax(outputs.logits, dim=-1)
            confidence, pred = torch.max(probs, dim=-1)

            pred_ids.append(pred.cpu().numpy())
            confidences.append(confidence.cpu().numpy())
            probs_all.append(probs.cpu().numpy())

    predicted_labels = np.concatenate(pred_ids, axis=0)
    confidence_scores = np.concatenate(confidences, axis=0)
    probability_matrix = np.concatenate(probs_all, axis=0)
    return predicted_labels, confidence_scores, probability_matrix


def build_predictions_dataframe(
    source_df: pd.DataFrame,
    predicted_labels: np.ndarray,
    confidence_scores: np.ndarray,
    label_names: dict[int, str],
) -> pd.DataFrame:
    if len(source_df) != len(predicted_labels):
        raise ValueError("Jumlah baris input tidak sama dengan jumlah hasil prediksi")

    if len(source_df) != len(confidence_scores):
        raise ValueError("Jumlah baris input tidak sama dengan jumlah confidence")

    result = source_df.copy()
    if "label_name" in result.columns:
        result = result.rename(columns={"label_name": "actual_label_name"})

    if "predicted_label" in result.columns:
        result = result.drop(columns=["predicted_label"])

    if "confidence" in result.columns:
        result = result.drop(columns=["confidence"])

    result["predicted_label"] = predicted_labels.astype(int)
    result["label_name"] = result["predicted_label"].map(label_names)
    result["confidence"] = confidence_scores.astype(float)

    if result["label_name"].isna().any():
        invalid_labels = sorted(result.loc[result["label_name"].isna(), "predicted_label"].unique().tolist())
        raise ValueError(f"Ditemukan label prediksi di luar mapping: {invalid_labels}")

    invalid_confidence = (~result["confidence"].between(0.0, 1.0)).sum()
    if invalid_confidence > 0:
        raise ValueError(f"Ditemukan {int(invalid_confidence)} nilai confidence di luar rentang 0-1")

    return result


def build_summary(
    config: InferenceConfig,
    input_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    probability_matrix: np.ndarray,
    label_names: dict[int, str],
) -> dict[str, Any]:
    counts = predictions_df["predicted_label"].value_counts().to_dict()
    distribution = {
        str(label): {
            "labelName": name,
            "count": int(counts.get(label, 0)),
        }
        for label, name in label_names.items()
    }

    return {
        "checkedAt": datetime.now().isoformat(timespec="seconds"),
        "config": asdict(config),
        "inputRows": int(len(input_df)),
        "outputRows": int(len(predictions_df)),
        "labelScheme": config.label_scheme,
        "textColumn": config.text_column,
        "hasAtColumn": "at" in predictions_df.columns,
        "predictionDistribution": distribution,
        "confidenceMin": float(predictions_df["confidence"].min()),
        "confidenceMax": float(predictions_df["confidence"].max()),
        "confidenceMean": float(predictions_df["confidence"].mean()),
        "probabilityShape": [int(probability_matrix.shape[0]), int(probability_matrix.shape[1])],
    }


def render_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_rows = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_row, separator_row, *body_rows])


def render_report(summary: dict[str, Any]) -> str:
    distribution_rows = []
    for label in sorted(summary["predictionDistribution"], key=int):
        item = summary["predictionDistribution"][label]
        distribution_rows.append([label, str(item["labelName"]), str(item["count"])])

    lines = [
        "# Laporan Inferensi Batch (Fase 09)",
        "",
        f"- Waktu inferensi: {summary['checkedAt']}",
        f"- Skema label: {summary['labelScheme']}",
        f"- Input inferensi: {summary['config']['input_path']}",
        f"- Output prediksi: {summary['config']['output_path']}",
        f"- Model dir: {summary['config']['model_dir']}",
        f"- Jumlah baris input: {summary['inputRows']}",
        f"- Jumlah baris output: {summary['outputRows']}",
        f"- Kolom teks: {summary['textColumn']}",
        f"- Kolom 'at' tersedia: {'Ya' if summary['hasAtColumn'] else 'Tidak'}",
        f"- Confidence min/mean/max: {summary['confidenceMin']:.4f} / {summary['confidenceMean']:.4f} / {summary['confidenceMax']:.4f}",
        "",
        "## Distribusi Prediksi",
        "",
        render_markdown_table(["Label", "Nama", "Jumlah"], distribution_rows),
        "",
    ]
    return "\n".join(lines)


def save_outputs(predictions_df: pd.DataFrame, summary: dict[str, Any], output_path: Path, summary_path: Path, report_path: Path) -> None:
    ensure_parent_directory(output_path)
    predictions_df.to_csv(output_path, index=False, encoding="utf-8")

    ensure_parent_directory(summary_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    ensure_parent_directory(report_path)
    report_path.write_text(render_report(summary), encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = to_config(args)
    logger = get_logger("predict_indobert", log_filename="predict_indobert.log")
    ensure_base_directories()

    input_path = Path(config.input_path)
    output_path = Path(config.output_path)
    summary_path = Path(config.summary_output)
    report_path = Path(config.report_output)
    model_dir = Path(config.model_dir)

    if not model_dir.exists():
        logger.error("Direktori model tidak ditemukan: %s", model_dir)
        return 1

    label_names = get_label_names(config.label_scheme)

    try:
        input_df = load_input_dataframe(input_path, config.text_column)
        predicted_labels, confidence_scores, probability_matrix = predict_sentiment(
            input_df[config.text_column].tolist(),
            model_path=model_dir,
            batch_size=config.batch_size,
            max_length=config.max_length,
        )
        predictions_df = build_predictions_dataframe(input_df, predicted_labels, confidence_scores, label_names)

        summary = build_summary(config, input_df, predictions_df, probability_matrix, label_names)
        save_outputs(predictions_df, summary, output_path, summary_path, report_path)
    except Exception as exc:
        logger.exception("Inferensi batch gagal: %s", exc)
        return 1

    logger.info("Inferensi batch selesai. Input=%s, Output=%s", len(input_df), len(predictions_df))
    logger.info("Predictions tersimpan di %s", output_path)
    logger.info("Ringkasan inferensi tersimpan di %s", summary_path)
    logger.info("Laporan inferensi tersimpan di %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
