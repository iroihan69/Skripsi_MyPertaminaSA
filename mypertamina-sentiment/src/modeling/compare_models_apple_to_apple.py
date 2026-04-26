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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import LOG_DIR, MODEL_OUTPUT_DIR, PROCESSED_DATA_DIR, ensure_base_directories
from src.utils.io_utils import ensure_parent_directory
from src.utils.logging_utils import get_logger
from src.utils.validation_utils import validate_required_columns

REQUIRED_COLUMNS = ["content_clean", "label"]


@dataclass(slots=True)
class CompareConfig:
    model_2class_dir: str
    model_3class_dir: str
    eval_input: str
    summary_output: str
    report_output: str
    batch_size: int
    max_length: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bandingkan model 2-kelas vs 3-kelas pada dataset yang sama (apple-to-apple)"
    )
    parser.add_argument(
        "--model-2class-dir",
        default=str(MODEL_OUTPUT_DIR / "baseline_indobert_2kelas_baru"),
        help="Direktori model 2-kelas",
    )
    parser.add_argument(
        "--model-3class-dir",
        default=str(MODEL_OUTPUT_DIR / "baseline_indobert_3kelas_baru"),
        help="Direktori model 3-kelas",
    )
    parser.add_argument(
        "--eval-input",
        default=str(PROCESSED_DATA_DIR / "test_data_3kelas.csv"),
        help="CSV evaluasi 3-kelas yang akan dipakai sebagai basis data sama",
    )
    parser.add_argument(
        "--summary-output",
        default=str(LOG_DIR / "evaluation_summary_apple_to_apple_2vs3_binary.json"),
        help="Output ringkasan JSON",
    )
    parser.add_argument(
        "--report-output",
        default=str(LOG_DIR / "evaluation_report_apple_to_apple_2vs3_binary.md"),
        help="Output laporan markdown",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size inferensi")
    parser.add_argument("--max-length", type=int, default=128, help="Panjang token maksimum")
    return parser.parse_args()


def to_config(args: argparse.Namespace) -> CompareConfig:
    return CompareConfig(
        model_2class_dir=str(Path(args.model_2class_dir).resolve()),
        model_3class_dir=str(Path(args.model_3class_dir).resolve()),
        eval_input=str(Path(args.eval_input).resolve()),
        summary_output=str(Path(args.summary_output).resolve()),
        report_output=str(Path(args.report_output).resolve()),
        batch_size=max(1, int(args.batch_size)),
        max_length=max(8, int(args.max_length)),
    )


def load_eval_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File evaluasi tidak ditemukan: {path}")

    df = pd.read_csv(path)
    missing_columns = validate_required_columns(df.columns, REQUIRED_COLUMNS)
    if missing_columns:
        raise ValueError(f"Kolom wajib belum lengkap: {', '.join(missing_columns)}")

    work_df = df.copy()
    work_df["content_clean"] = work_df["content_clean"].fillna("").astype(str)
    work_df["label"] = pd.to_numeric(work_df["label"], errors="coerce")

    if work_df["label"].isna().any():
        raise ValueError("Ditemukan label non-numerik pada file evaluasi")

    work_df["label"] = work_df["label"].astype(int)
    invalid = sorted(set(work_df["label"].tolist()) - {0, 1, 2})
    if invalid:
        raise ValueError(f"Label di luar skema 3-kelas ditemukan: {invalid}")

    blank_count = int(work_df["content_clean"].str.strip().eq("").sum())
    if blank_count > 0:
        raise ValueError(f"Ditemukan content_clean kosong sebanyak {blank_count} baris")

    return work_df.reset_index(drop=True)


def predict_labels(
    model_dir: Path,
    texts: list[str],
    *,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    pred_ids: list[np.ndarray] = []
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
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            pred = torch.argmax(outputs.logits, dim=-1)
            pred_ids.append(pred.cpu().numpy())

    return np.concatenate(pred_ids, axis=0).astype(int)


def calculate_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Negatif", "Positif"],
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "confusion_matrix_binary": {
            "Aktual Negatif": {
                "Prediksi Negatif": int(cm[0, 0]),
                "Prediksi Positif": int(cm[0, 1]),
            },
            "Aktual Positif": {
                "Prediksi Negatif": int(cm[1, 0]),
                "Prediksi Positif": int(cm[1, 1]),
            },
        },
        "classification_report_binary": report,
    }


def render_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_rows = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_row, separator_row, *body_rows])


def render_report(summary: dict[str, Any]) -> str:
    m2 = summary["results"]["model_2class"]
    m3 = summary["results"]["model_3class_mapped_to_2class"]
    diff_acc = m2["accuracy"] - m3["accuracy"]
    diff_f1 = m2["weighted_f1"] - m3["weighted_f1"]

    metrics_rows = [
        ["Model 2-kelas", f"{m2['accuracy']:.4f}", f"{m2['weighted_f1']:.4f}", f"{m2['macro_f1']:.4f}"],
        ["Model 3-kelas (map ke 2-kelas)", f"{m3['accuracy']:.4f}", f"{m3['weighted_f1']:.4f}", f"{m3['macro_f1']:.4f}"],
        ["Selisih (2-kelas - 3-kelas-map)", f"{diff_acc:+.4f}", f"{diff_f1:+.4f}", "-"],
    ]

    lines = [
        "# Laporan Perbandingan Apple-to-Apple (2-kelas vs 3-kelas)",
        "",
        f"- Waktu evaluasi: {summary['checkedAt']}",
        f"- Dataset evaluasi bersama: {summary['evalInput']}",
        "- Dasar pembandingan: kedua model diprediksi pada baris data yang sama.",
        "- Aturan mapping biner: label Positif(2)->1; selain itu ->0.",
        "- Catatan: pada skema ini, kelas Netral dari data 3-kelas digabung ke Negatif sesuai aturan 2-kelas proyek.",
        "",
        "## Ringkasan Metrik",
        "",
        render_markdown_table(["Model", "Accuracy", "Weighted F1", "Macro F1"], metrics_rows),
        "",
        "## Kesimpulan Singkat",
        "",
        (
            "- Model 2-kelas unggul pada skema pembandingan biner yang sama."
            if diff_f1 > 0
            else "- Model 3-kelas (setelah dipetakan ke biner) unggul pada skema pembandingan biner yang sama."
        ),
        "- Hasil ini memisahkan efek model dari efek perbedaan dataset evaluasi sebelumnya.",
    ]
    return "\n".join(lines)


def main() -> None:
    ensure_base_directories()
    logger = get_logger(__name__)

    config = to_config(parse_args())
    logger.info("Memulai perbandingan apple-to-apple")

    eval_df = load_eval_dataframe(Path(config.eval_input))
    texts = eval_df["content_clean"].tolist()

    # Ground truth biner dari label 3-kelas: Positif(2) -> 1, lainnya -> 0.
    y_true_binary = (eval_df["label"].to_numpy() == 2).astype(int)

    preds_2class = predict_labels(
        Path(config.model_2class_dir),
        texts,
        batch_size=config.batch_size,
        max_length=config.max_length,
    )
    invalid_2 = sorted(set(preds_2class.tolist()) - {0, 1})
    if invalid_2:
        raise ValueError(f"Prediksi model 2-kelas di luar rentang 0/1: {invalid_2}")

    preds_3class = predict_labels(
        Path(config.model_3class_dir),
        texts,
        batch_size=config.batch_size,
        max_length=config.max_length,
    )
    invalid_3 = sorted(set(preds_3class.tolist()) - {0, 1, 2})
    if invalid_3:
        raise ValueError(f"Prediksi model 3-kelas di luar rentang 0/1/2: {invalid_3}")

    preds_3class_binary = (preds_3class == 2).astype(int)

    metrics_2class = calculate_binary_metrics(y_true_binary, preds_2class)
    metrics_3class_binary = calculate_binary_metrics(y_true_binary, preds_3class_binary)

    summary = {
        "checkedAt": datetime.now().isoformat(timespec="seconds"),
        "evalInput": config.eval_input,
        "comparisonType": "apple_to_apple_binary_on_same_dataset",
        "mappingRule": {
            "groundTruth": "label==2 -> Positif(1), label in {0,1} -> Negatif(0)",
            "predictions3Class": "pred==2 -> Positif(1), pred in {0,1} -> Negatif(0)",
        },
        "config": asdict(config),
        "results": {
            "model_2class": metrics_2class,
            "model_3class_mapped_to_2class": metrics_3class_binary,
        },
    }

    summary_output = Path(config.summary_output)
    report_output = Path(config.report_output)
    ensure_parent_directory(summary_output)
    ensure_parent_directory(report_output)

    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_output.write_text(render_report(summary), encoding="utf-8")

    logger.info("Selesai. Ringkasan: %s", summary_output)
    logger.info("Selesai. Laporan : %s", report_output)


if __name__ == "__main__":
    main()
