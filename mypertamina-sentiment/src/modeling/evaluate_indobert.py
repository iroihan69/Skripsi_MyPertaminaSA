from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import (
    LOG_DIR,
    MODEL_OUTPUT_DIR,
    PROCESSED_DATA_DIR,
    TEST_DATA_FILENAME,
    ensure_base_directories,
)
from src.utils.io_utils import ensure_parent_directory
from src.utils.logging_utils import get_logger
from src.utils.validation_utils import validate_required_columns

REQUIRED_COLUMNS = ["content_clean", "label"]
LABEL_NAMES_3CLASS = {0: "Negatif", 1: "Netral", 2: "Positif"}
LABEL_NAMES_2CLASS = {0: "Negatif", 1: "Positif"}


def get_label_names(label_scheme: str) -> dict[int, str]:
    if label_scheme == "2class":
        return LABEL_NAMES_2CLASS
    return LABEL_NAMES_3CLASS


@dataclass(slots=True)
class EvaluationConfig:
    model_dir: str
    eval_input: str
    report_output: str
    summary_output: str
    confusion_matrix_png: str
    misclassified_output: str
    top_error_pairs: int
    max_error_samples: int
    decision_weighted_f1_threshold: float
    label_scheme: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluasi baseline IndoBERT untuk Fase 08")
    parser.add_argument(
        "--model-dir",
        default=str(MODEL_OUTPUT_DIR / "baseline_indobert"),
        help="Direktori model hasil training (folder final atau checkpoint)",
    )
    parser.add_argument(
        "--eval-input",
        default=str(PROCESSED_DATA_DIR / TEST_DATA_FILENAME),
        help="Path CSV test_data.csv",
    )
    parser.add_argument(
        "--report-output",
        default=str(LOG_DIR / "evaluation_report.md"),
        help="Path output markdown laporan evaluasi",
    )
    parser.add_argument(
        "--summary-output",
        default=str(LOG_DIR / "evaluation_summary.json"),
        help="Path output JSON ringkasan evaluasi",
    )
    parser.add_argument(
        "--confusion-matrix-png",
        default=str(LOG_DIR / "evaluation_confusion_matrix.png"),
        help="Path output PNG confusion matrix",
    )
    parser.add_argument(
        "--misclassified-output",
        default=str(LOG_DIR / "evaluation_misclassified_samples.csv"),
        help="Path output CSV sampel salah prediksi",
    )
    parser.add_argument(
        "--top-error-pairs",
        type=int,
        default=3,
        help="Jumlah pasangan kelas tertukar teratas yang diringkas",
    )
    parser.add_argument(
        "--max-error-samples",
        type=int,
        default=30,
        help="Jumlah sampel salah prediksi yang disimpan ke CSV",
    )
    parser.add_argument(
        "--decision-weighted-f1-threshold",
        type=float,
        default=0.80,
        help="Ambang weighted F1 untuk keputusan gate baseline",
    )
    parser.add_argument(
        "--label-scheme",
        choices=["2class", "3class"],
        default="3class",
        help="Skema label dataset evaluasi",
    )
    return parser.parse_args()


def to_config(args: argparse.Namespace) -> EvaluationConfig:
    return EvaluationConfig(
        model_dir=str(Path(args.model_dir).resolve()),
        eval_input=str(Path(args.eval_input).resolve()),
        report_output=str(Path(args.report_output).resolve()),
        summary_output=str(Path(args.summary_output).resolve()),
        confusion_matrix_png=str(Path(args.confusion_matrix_png).resolve()),
        misclassified_output=str(Path(args.misclassified_output).resolve()),
        top_error_pairs=max(1, int(args.top_error_pairs)),
        max_error_samples=max(1, int(args.max_error_samples)),
        decision_weighted_f1_threshold=float(args.decision_weighted_f1_threshold),
        label_scheme=args.label_scheme,
    )


def load_eval_dataframe(path: Path, label_names: dict[int, str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File dataset evaluasi tidak ditemukan: {path}")

    dataframe = pd.read_csv(path)
    missing_columns = validate_required_columns(dataframe.columns, REQUIRED_COLUMNS)
    if missing_columns:
        raise ValueError(f"Kolom wajib dataset evaluasi belum lengkap: {', '.join(missing_columns)}")

    work_df = dataframe.copy()
    work_df["content_clean"] = work_df["content_clean"].fillna("").astype(str)
    work_df["label"] = pd.to_numeric(work_df["label"], errors="coerce")

    blank_count = int(work_df["content_clean"].str.strip().eq("").sum())
    if blank_count > 0:
        raise ValueError(f"Ditemukan content_clean kosong sebanyak {blank_count} baris pada {path}")

    if work_df["label"].isna().any():
        raise ValueError(f"Terdapat label non-numerik pada dataset evaluasi: {path}")

    work_df["label"] = work_df["label"].astype(int)
    invalid_labels = sorted(set(work_df["label"].tolist()) - set(label_names))
    if invalid_labels:
        expected_label_text = "/".join(str(label) for label in sorted(label_names))
        raise ValueError(f"Ditemukan label di luar rentang {expected_label_text}: {invalid_labels}")

    return work_df.reset_index(drop=True)


def predict_labels(model_dir: Path, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    batch_size = 32
    logits_list: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            batch = tokenizer(
                batch_texts,
                truncation=True,
                max_length=128,
                padding=True,
                return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            logits_list.append(outputs.logits.cpu().numpy())

    logits = np.concatenate(logits_list, axis=0)
    predictions = np.argmax(logits, axis=1)
    probabilities = torch.softmax(torch.tensor(logits), dim=1).numpy()
    return predictions, probabilities


def build_confusion_matrix_dataframe(y_true: np.ndarray, y_pred: np.ndarray, label_names: dict[int, str]) -> pd.DataFrame:
    label_order = sorted(label_names)
    cm = confusion_matrix(y_true, y_pred, labels=label_order)
    row_labels = [f"Aktual {label_names[label]}" for label in label_order]
    col_labels = [f"Prediksi {label_names[label]}" for label in label_order]
    return pd.DataFrame(cm, index=row_labels, columns=col_labels)


def save_confusion_matrix_png(cm_df: pd.DataFrame, output_path: Path) -> None:
    ensure_parent_directory(output_path)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_df,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )
    ax.set_title("Confusion Matrix Baseline IndoBERT")
    ax.set_xlabel("Label Prediksi")
    ax.set_ylabel("Label Aktual")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_error_pairs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    top_k: int,
    label_names: dict[int, str],
) -> list[dict[str, Any]]:
    confusion_pairs = Counter()
    for true_label, pred_label in zip(y_true.tolist(), y_pred.tolist()):
        if true_label != pred_label:
            confusion_pairs[(true_label, pred_label)] += 1

    top_pairs: list[dict[str, Any]] = []
    for (true_label, pred_label), count in confusion_pairs.most_common(top_k):
        top_pairs.append(
            {
                "actualLabelId": int(true_label),
                "actualLabel": label_names[int(true_label)],
                "predictedLabelId": int(pred_label),
                "predictedLabel": label_names[int(pred_label)],
                "count": int(count),
            }
        )
    return top_pairs


def build_misclassified_samples(
    eval_df: pd.DataFrame,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    max_samples: int,
    label_names: dict[int, str],
) -> pd.DataFrame:
    work_df = eval_df.copy()
    work_df["pred_label"] = y_pred
    work_df["pred_label_name"] = work_df["pred_label"].map(label_names)
    work_df["confidence"] = probabilities.max(axis=1)

    mis_df = work_df[work_df["label"] != work_df["pred_label"]].copy()
    mis_df["label_name"] = mis_df["label"].map(label_names)

    keep_columns = [
        "reviewId",
        "content",
        "content_clean",
        "label",
        "label_name",
        "pred_label",
        "pred_label_name",
        "confidence",
        "score",
        "at",
    ]
    existing_columns = [col for col in keep_columns if col in mis_df.columns]
    result = mis_df.sort_values("confidence", ascending=False).head(max_samples).loc[:, existing_columns]
    return result.reset_index(drop=True)


def build_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def render_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_rows = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_row, separator_row, *body_rows])


def render_report(summary: dict[str, Any]) -> str:
    metric_rows = [[name, f"{value:.4f}"] for name, value in summary["metrics"].items()]

    cm_rows: list[list[str]] = []
    cm_dict = summary["confusionMatrix"]
    for actual_label, pred_map in cm_dict.items():
        cm_rows.append([actual_label, *[str(value) for value in pred_map.values()]])

    class_rows = []
    for label_name in summary["labelOrderNames"]:
        values = summary["classificationReport"][label_name]
        class_rows.append(
            [
                label_name,
                f"{values['precision']:.4f}",
                f"{values['recall']:.4f}",
                f"{values['f1-score']:.4f}",
                str(int(values['support'])),
            ]
        )

    top_error_rows = []
    for pair in summary["topConfusions"]:
        top_error_rows.append(
            [
                pair["actualLabel"],
                pair["predictedLabel"],
                str(pair["count"]),
            ]
        )

    if not top_error_rows:
        top_error_rows = [["-", "-", "0"]]

    lines = [
        "# Laporan Evaluasi Model (Fase 08)",
        "",
        f"- Waktu evaluasi: {summary['checkedAt']}",
        f"- Skema label: {summary['labelScheme']}",
        f"- Model dievaluasi: {summary['modelDir']}",
        f"- Dataset evaluasi: {summary['evalInput']}",
        f"- Jumlah baris evaluasi: {summary['evalRows']}",
        f"- Ambang keputusan weighted F1: {summary['decisionThreshold']:.2f}",
        f"- Keputusan baseline: **{summary['decision']}**",
        "",
        "## Metrik Utama",
        "",
        render_markdown_table(["Metrik", "Nilai"], metric_rows),
        "",
        "## Confusion Matrix (Tabel)",
        "",
        render_markdown_table(
            ["Aktual", *summary["predictedHeaders"]],
            cm_rows,
        ),
        "",
        "## Classification Report per Kelas",
        "",
        render_markdown_table(["Kelas", "Precision", "Recall", "F1", "Support"], class_rows),
        "",
        "## Error Analysis Ringkas",
        "",
        f"- Jumlah salah prediksi: {summary['misclassifiedCount']}",
        f"- Sampel salah prediksi tersimpan di: {summary['misclassifiedOutput']}",
        f"- Confusion matrix PNG tersimpan di: {summary['confusionMatrixPng']}",
        "",
        "Pasangan kelas yang paling sering tertukar:",
        "",
        render_markdown_table(["Aktual", "Sering Diprediksi Sebagai", "Jumlah"], top_error_rows),
        "",
        "Interpretasi singkat:",
        f"- Weighted F1 berada pada {summary['metrics']['weighted_f1']:.4f}.",
        "- Evaluasi menunjukkan pemisahan kelas minoritas perlu dipantau pada iterasi berikutnya.",
        "- Dengan keputusan default proyek, baseline tetap diterima untuk lanjut Fase 09 jika weighted F1 memenuhi ambang.",
        "",
    ]
    return "\n".join(lines)


def save_outputs(
    summary: dict[str, Any],
    report_output: Path,
    summary_output: Path,
    misclassified_df: pd.DataFrame,
    misclassified_output: Path,
) -> None:
    ensure_parent_directory(report_output)
    report_output.write_text(render_report(summary), encoding="utf-8")

    ensure_parent_directory(summary_output)
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    ensure_parent_directory(misclassified_output)
    misclassified_df.to_csv(misclassified_output, index=False)


def main() -> int:
    args = parse_args()
    config = to_config(args)
    logger = get_logger("evaluate_indobert", log_filename="evaluation.log")

    ensure_base_directories()

    model_dir = Path(config.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Direktori model tidak ditemukan: {model_dir}")

    label_names = get_label_names(config.label_scheme)
    eval_df = load_eval_dataframe(Path(config.eval_input), label_names)
    y_true = eval_df["label"].to_numpy()

    logger.info("Memulai evaluasi model pada %s", model_dir)
    y_pred, probabilities = predict_labels(model_dir=model_dir, texts=eval_df["content_clean"].tolist())

    metrics = build_metrics(y_true=y_true, y_pred=y_pred)
    class_report = classification_report(
        y_true,
        y_pred,
        labels=sorted(label_names),
        target_names=[label_names[label] for label in sorted(label_names)],
        zero_division=0,
        output_dict=True,
    )

    cm_df = build_confusion_matrix_dataframe(y_true=y_true, y_pred=y_pred, label_names=label_names)
    save_confusion_matrix_png(cm_df=cm_df, output_path=Path(config.confusion_matrix_png))

    top_confusions = build_error_pairs(y_true=y_true, y_pred=y_pred, top_k=config.top_error_pairs, label_names=label_names)
    misclassified_df = build_misclassified_samples(
        eval_df=eval_df,
        y_pred=y_pred,
        probabilities=probabilities,
        max_samples=config.max_error_samples,
        label_names=label_names,
    )

    decision = "Terima baseline dan lanjut Fase 09"
    if metrics["weighted_f1"] < config.decision_weighted_f1_threshold:
        decision = "Perlu tuning ulang sebelum lanjut Fase 09"

    summary = {
        "checkedAt": datetime.now().isoformat(timespec="seconds"),
        "modelDir": str(model_dir),
        "evalInput": config.eval_input,
        "evalRows": int(len(eval_df)),
        "decisionThreshold": float(config.decision_weighted_f1_threshold),
        "decision": decision,
        "labelScheme": config.label_scheme,
        "labelOrderNames": [label_names[label] for label in sorted(label_names)],
        "predictedHeaders": [f"Prediksi {label_names[label]}" for label in sorted(label_names)],
        "metrics": metrics,
        "confusionMatrix": cm_df.to_dict(orient="index"),
        "classificationReport": {label_names[label]: class_report[label_names[label]] for label in sorted(label_names)},
        "topConfusions": top_confusions,
        "misclassifiedCount": int((y_true != y_pred).sum()),
        "misclassifiedOutput": config.misclassified_output,
        "confusionMatrixPng": config.confusion_matrix_png,
        "labelDistributionEval": dict(Counter(y_true.tolist())),
        "predictedDistribution": dict(Counter(y_pred.tolist())),
        "config": asdict(config),
    }

    save_outputs(
        summary=summary,
        report_output=Path(config.report_output),
        summary_output=Path(config.summary_output),
        misclassified_df=misclassified_df,
        misclassified_output=Path(config.misclassified_output),
    )

    logger.info("Evaluasi selesai. Weighted F1: %.4f", metrics["weighted_f1"])
    logger.info("Keputusan baseline: %s", decision)
    logger.info("Laporan evaluasi tersimpan di %s", config.report_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
