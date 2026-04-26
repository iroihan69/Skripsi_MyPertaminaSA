from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.config import (
    LOG_DIR,
    MODEL_NAME,
    MODEL_OUTPUT_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    TEST_DATA_FILENAME,
    TRAIN_DATA_FILENAME,
    ensure_base_directories,
)
from src.utils.io_utils import ensure_directory, ensure_parent_directory
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
class TrainingConfig:
    train_path: str
    eval_path: str
    output_dir: str
    report_output: str
    summary_output: str
    log_filename: str
    model_name: str
    max_length: int
    num_train_epochs: float
    train_batch_size: int
    eval_batch_size: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    gradient_accumulation_steps: int
    logging_steps: int
    eval_steps: int
    save_steps: int
    early_stopping_patience: int
    metric_for_best_model: str
    random_seed: int
    label_scheme: str
    use_class_weighting: bool


class ReviewDataset(Dataset):
    def __init__(self, encodings: dict[str, list[list[int]]], labels: list[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: torch.tensor(value[index], dtype=torch.long) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


class WeightedTrainer(Trainer):
    def __init__(self, *args: Any, class_weights: torch.Tensor | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> Any:
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_weights = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_fct = torch.nn.CrossEntropyLoss(weight=loss_weights)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tuning baseline IndoBERT untuk klasifikasi sentimen MyPertamina")
    parser.add_argument(
        "--train-input",
        default=str(PROCESSED_DATA_DIR / TRAIN_DATA_FILENAME),
        help="Path CSV train_data.csv",
    )
    parser.add_argument(
        "--eval-input",
        default=str(PROCESSED_DATA_DIR / TEST_DATA_FILENAME),
        help="Path CSV test_data.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=str(MODEL_OUTPUT_DIR / "baseline_indobert"),
        help="Direktori artefak model hasil training",
    )
    parser.add_argument(
        "--report-output",
        default=str(LOG_DIR / "training_report.md"),
        help="Path output markdown laporan training",
    )
    parser.add_argument(
        "--summary-output",
        default=str(LOG_DIR / "training_summary.json"),
        help="Path output JSON ringkasan training",
    )
    parser.add_argument(
        "--log-filename",
        default="training.log",
        help="Nama file log training di folder logs",
    )
    parser.add_argument(
        "--model-name",
        default=MODEL_NAME,
        help="Nama model Hugging Face yang dipakai",
    )
    parser.add_argument("--max-length", type=int, default=128, help="Panjang token maksimum")
    parser.add_argument("--epochs", type=float, default=3.0, help="Jumlah epoch training")
    parser.add_argument("--train-batch-size", type=int, default=8, help="Batch size train per device")
    parser.add_argument("--eval-batch-size", type=int, default=16, help="Batch size evaluasi per device")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate AdamW")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Akumulasi gradien untuk simulasi effective batch lebih besar",
    )
    parser.add_argument("--logging-steps", type=int, default=25, help="Interval langkah logging")
    parser.add_argument("--eval-steps", type=int, default=50, help="Interval langkah evaluasi")
    parser.add_argument("--save-steps", type=int, default=50, help="Interval langkah simpan checkpoint")
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=2,
        help="Jumlah evaluasi berturut-turut tanpa perbaikan sebelum early stopping",
    )
    parser.add_argument(
        "--metric-for-best-model",
        default="macro_f1",
        choices=["weighted_f1", "macro_f1", "accuracy"],
        help="Metrik pemilihan model terbaik",
    )
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED, help="Seed random global")
    parser.add_argument(
        "--label-scheme",
        choices=["2class", "3class"],
        default="3class",
        help="Skema label dataset yang dipakai saat training",
    )
    parser.add_argument(
        "--use-class-weighting",
        action="store_true",
        help="Aktifkan class weighting pada loss training",
    )
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(path: Path, label_names: dict[int, str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File dataset tidak ditemukan: {path}")

    dataframe = pd.read_csv(path)
    missing_columns = validate_required_columns(dataframe.columns, REQUIRED_COLUMNS)
    if missing_columns:
        raise ValueError(f"Kolom wajib dataset belum lengkap: {', '.join(missing_columns)}")

    work_df = dataframe.copy()
    work_df["content_clean"] = work_df["content_clean"].fillna("").astype(str)
    work_df["label"] = pd.to_numeric(work_df["label"], errors="coerce")

    blank_count = int(work_df["content_clean"].str.strip().eq("").sum())
    if blank_count > 0:
        raise ValueError(f"Ditemukan content_clean kosong sebanyak {blank_count} baris pada {path}")

    if work_df["label"].isna().any():
        raise ValueError(f"Terdapat label non-numerik pada dataset {path}")

    work_df["label"] = work_df["label"].astype(int)
    invalid_labels = sorted(set(work_df["label"].tolist()) - set(label_names))
    if invalid_labels:
        expected_label_text = "/".join(str(label) for label in sorted(label_names))
        raise ValueError(f"Ditemukan label di luar rentang {expected_label_text}: {invalid_labels}")

    return work_df.reset_index(drop=True)


def build_distribution(labels: list[int], label_names: dict[int, str]) -> dict[str, dict[str, float | int | str]]:
    counts = Counter(int(label) for label in labels)
    total = sum(counts.values())
    distribution: dict[str, dict[str, float | int | str]] = {}
    for label_id, label_name in label_names.items():
        count = int(counts.get(label_id, 0))
        percentage = (count / total) * 100 if total else 0.0
        distribution[str(label_id)] = {
            "labelName": label_name,
            "count": count,
            "percentage": round(percentage, 2),
        }
    return distribution


def build_class_weights(labels: list[int], label_names: dict[int, str]) -> dict[int, float]:
    counts = Counter(int(label) for label in labels)
    total = sum(counts.values())
    num_classes = len(label_names)

    if total == 0:
        raise ValueError("Dataset train kosong, class weighting tidak dapat dihitung")

    class_weights: dict[int, float] = {}
    for label_id in sorted(label_names):
        count = int(counts.get(label_id, 0))
        if count <= 0:
            raise ValueError(f"Kelas {label_id} tidak ditemukan di train set, class weighting tidak dapat dihitung")
        class_weights[label_id] = float(total / (num_classes * count))

    return class_weights


def tokenize_texts(tokenizer: AutoTokenizer, texts: list[str], max_length: int) -> dict[str, list[list[int]]]:
    return tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
    )


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
    }


def sanitize_history(log_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned_history: list[dict[str, Any]] = []
    for item in log_history:
        normalized_item: dict[str, Any] = {}
        for key, value in item.items():
            if isinstance(value, (np.floating, np.integer)):
                normalized_item[key] = value.item()
            else:
                normalized_item[key] = value
        cleaned_history.append(normalized_item)
    return cleaned_history


def build_summary(
    config: TrainingConfig,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    device: str,
    train_result: Any,
    eval_metrics: dict[str, float],
    trainer: Trainer,
    label_names: dict[int, str],
    class_weights: dict[int, float] | None,
) -> dict[str, Any]:
    train_runtime = float(train_result.metrics.get("train_runtime", 0.0))
    train_loss = float(train_result.metrics.get("train_loss", 0.0))
    effective_batch_size = config.train_batch_size * config.gradient_accumulation_steps
    best_metric_value = float(getattr(trainer.state, "best_metric", 0.0) or 0.0)
    best_checkpoint = getattr(trainer.state, "best_model_checkpoint", None)

    return {
        "checkedAt": datetime.now().isoformat(timespec="seconds"),
        "trainInputPath": config.train_path,
        "evalInputPath": config.eval_path,
        "outputDir": config.output_dir,
        "reportOutput": config.report_output,
        "summaryOutput": config.summary_output,
        "modelName": config.model_name,
        "device": device,
        "randomSeed": config.random_seed,
        "maxLength": config.max_length,
        "numTrainEpochs": config.num_train_epochs,
        "learningRate": config.learning_rate,
        "weightDecay": config.weight_decay,
        "warmupRatio": config.warmup_ratio,
        "trainBatchSize": config.train_batch_size,
        "evalBatchSize": config.eval_batch_size,
        "gradientAccumulationSteps": config.gradient_accumulation_steps,
        "effectiveTrainBatchSize": effective_batch_size,
        "metricForBestModel": config.metric_for_best_model,
        "labelScheme": config.label_scheme,
        "useClassWeighting": config.use_class_weighting,
        "classWeights": {str(label): float(weight) for label, weight in class_weights.items()} if class_weights else None,
        "trainRows": int(len(train_df)),
        "evalRows": int(len(eval_df)),
        "trainDistribution": build_distribution(train_df["label"].tolist(), label_names),
        "evalDistribution": build_distribution(eval_df["label"].tolist(), label_names),
        "trainRuntimeSeconds": train_runtime,
        "trainLoss": train_loss,
        "bestMetric": best_metric_value,
        "bestCheckpoint": best_checkpoint,
        "evalMetrics": {key: float(value) for key, value in eval_metrics.items()},
        "logHistory": sanitize_history(trainer.state.log_history),
    }


def render_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_rows = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_row, separator_row, *body_rows])


def render_report(summary: dict[str, Any]) -> str:
    hyperparameter_rows = [
        ["Model", str(summary["modelName"])],
        ["Device", str(summary["device"])],
        ["Epoch", str(summary["numTrainEpochs"])],
        ["Max length", str(summary["maxLength"])],
        ["Learning rate", str(summary["learningRate"])],
        ["Weight decay", str(summary["weightDecay"])],
        ["Warmup ratio", str(summary["warmupRatio"])],
        ["Train batch size", str(summary["trainBatchSize"])],
        ["Gradient accumulation", str(summary["gradientAccumulationSteps"])],
        ["Effective batch size", str(summary["effectiveTrainBatchSize"])],
        ["Eval batch size", str(summary["evalBatchSize"])],
        ["Random seed", str(summary["randomSeed"])],
        ["Metric best model", str(summary["metricForBestModel"])],
        ["Class weighting", "Aktif" if summary["useClassWeighting"] else "Nonaktif"],
    ]
    distribution_rows: list[list[str]] = []
    for split_name in ["trainDistribution", "evalDistribution"]:
        split_label = "Train" if split_name == "trainDistribution" else "Eval"
        for label_id, info in summary[split_name].items():
            distribution_rows.append(
                [
                    split_label,
                    label_id,
                    str(info["labelName"]),
                    str(info["count"]),
                    f"{float(info['percentage']):.2f}%",
                ]
            )
    metric_rows = [[key, f"{float(value):.4f}"] for key, value in summary["evalMetrics"].items()]

    lines = [
        "# Laporan Fine-tuning IndoBERT",
        "",
        f"- Waktu proses: {summary['checkedAt']}",
        f"- Skema label: {summary['labelScheme']}",
        f"- Train rows: {summary['trainRows']}",
        f"- Eval rows: {summary['evalRows']}",
        f"- Runtime training (detik): {summary['trainRuntimeSeconds']:.2f}",
        f"- Train loss akhir: {summary['trainLoss']:.4f}",
        f"- Best checkpoint: {summary['bestCheckpoint'] or '-'}",
        f"- Best metric ({summary['metricForBestModel']}): {summary['bestMetric']:.4f}",
        "",
        "## Hyperparameter Baseline",
        "",
        render_markdown_table(["Parameter", "Nilai"], hyperparameter_rows),
        "",
        "## Distribusi Dataset",
        "",
        render_markdown_table(["Split", "Label", "Nama", "Jumlah", "Persentase"], distribution_rows),
        "",
        "## Bobot Kelas",
        "",
        render_markdown_table(
            ["Label", "Nama", "Bobot"],
            [
                [label_id, str(info["labelName"]), f"{float(summary['classWeights'][label_id]):.4f}"]
                for label_id, info in summary["trainDistribution"].items()
                if summary["classWeights"] is not None and label_id in summary["classWeights"]
            ]
            if summary["classWeights"] is not None
            else [["-", "-", "-"]],
        ),
        "",
        "## Hasil Evaluasi",
        "",
        render_markdown_table(["Metrik", "Nilai"], metric_rows),
        "",
    ]
    return "\n".join(lines)


def save_outputs(summary: dict[str, Any], report_output: Path, summary_output: Path) -> None:
    ensure_parent_directory(report_output)
    report_output.write_text(render_report(summary), encoding="utf-8")

    ensure_parent_directory(summary_output)
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def to_training_config(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        train_path=str(Path(args.train_input).resolve()),
        eval_path=str(Path(args.eval_input).resolve()),
        output_dir=str(Path(args.output_dir).resolve()),
        report_output=str(Path(args.report_output).resolve()),
        summary_output=str(Path(args.summary_output).resolve()),
        log_filename=str(args.log_filename),
        model_name=args.model_name,
        max_length=args.max_length,
        num_train_epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        early_stopping_patience=args.early_stopping_patience,
        metric_for_best_model=args.metric_for_best_model,
        random_seed=args.random_seed,
        label_scheme=args.label_scheme,
        use_class_weighting=bool(args.use_class_weighting),
    )


def main() -> int:
    args = parse_args()
    config = to_training_config(args)
    logger = get_logger("train_indobert", log_filename=config.log_filename)

    ensure_base_directories()
    ensure_directory(config.output_dir)
    set_global_seed(config.random_seed)

    logger.info("Memulai fine-tuning IndoBERT dengan config: %s", json.dumps(asdict(config), ensure_ascii=False))

    label_names = get_label_names(config.label_scheme)
    train_df = load_dataset(Path(config.train_path), label_names)
    eval_df = load_dataset(Path(config.eval_path), label_names)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=len(label_names))

    train_encodings = tokenize_texts(tokenizer, train_df["content_clean"].tolist(), max_length=config.max_length)
    eval_encodings = tokenize_texts(tokenizer, eval_df["content_clean"].tolist(), max_length=config.max_length)

    train_dataset = ReviewDataset(train_encodings, train_df["label"].tolist())
    eval_dataset = ReviewDataset(eval_encodings, eval_df["label"].tolist())
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    class_weights = build_class_weights(train_df["label"].tolist(), label_names) if config.use_class_weighting else None
    class_weights_tensor = (
        torch.tensor([class_weights[label_id] for label_id in sorted(label_names)], dtype=torch.float32)
        if class_weights is not None
        else None
    )

    # Filter argumen agar TrainingArguments tetap kompatibel pada variasi versi transformers.
    training_args_kwargs: dict[str, Any] = {
        "output_dir": config.output_dir,
        "overwrite_output_dir": True,
        "do_train": True,
        "do_eval": True,
        "eval_strategy": "steps",
        "save_strategy": "steps",
        "logging_strategy": "steps",
        "eval_steps": config.eval_steps,
        "save_steps": config.save_steps,
        "logging_steps": config.logging_steps,
        "num_train_epochs": config.num_train_epochs,
        "per_device_train_batch_size": config.train_batch_size,
        "per_device_eval_batch_size": config.eval_batch_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "warmup_ratio": config.warmup_ratio,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "seed": config.random_seed,
        "load_best_model_at_end": True,
        "metric_for_best_model": config.metric_for_best_model,
        "greater_is_better": True,
        "save_total_limit": 2,
        "report_to": [],
        "logging_dir": str(Path(config.output_dir) / "runs"),
        "fp16": torch.cuda.is_available(),
        "dataloader_num_workers": 0,
        "remove_unused_columns": False,
    }

    supported_training_args = set(inspect.signature(TrainingArguments.__init__).parameters)
    filtered_training_args_kwargs = {
        key: value for key, value in training_args_kwargs.items() if key in supported_training_args
    }
    training_args = TrainingArguments(**filtered_training_args_kwargs)

    trainer_cls = WeightedTrainer if class_weights_tensor is not None else Trainer
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "tokenizer": tokenizer,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)],
    }
    if class_weights_tensor is not None:
        trainer_kwargs["class_weights"] = class_weights_tensor
    trainer = trainer_cls(**trainer_kwargs)

    train_result = trainer.train()
    eval_metrics = trainer.evaluate(eval_dataset=eval_dataset)
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    summary = build_summary(
        config,
        train_df,
        eval_df,
        device,
        train_result,
        eval_metrics,
        trainer,
        label_names,
        class_weights,
    )
    save_outputs(summary, Path(config.report_output), Path(config.summary_output))

    logger.info("Training selesai. Weighted F1 evaluasi: %.4f", summary["evalMetrics"].get("eval_weighted_f1", 0.0))
    logger.info("Artefak model tersimpan di %s", config.output_dir)
    logger.info("Ringkasan training tersimpan di %s", config.summary_output)
    logger.info("Laporan training tersimpan di %s", config.report_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())