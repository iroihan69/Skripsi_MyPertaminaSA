from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

from src.config import (
    LOG_DIR,
    MODEL_OUTPUT_DIR,
    PREPROCESSED_REVIEWS_FILENAME,
    PROCESSED_DATA_DIR,
    ensure_base_directories,
)
from src.modeling.predict_indobert import LABEL_NAMES, predict_sentiment
from src.utils.io_utils import ensure_parent_directory
from src.utils.logging_utils import get_logger


@dataclass(slots=True)
class Rating3WordcloudConfig:
    input_path: str
    model_dir: str
    output_dir: str
    text_column: str
    score_column: str
    date_column: str
    rating_value: int
    batch_size: int
    max_length: int
    min_words_for_wordcloud: int
    max_words: int
    width: int
    height: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analisis wordcloud rating 3 (overall + per tahun) dengan inferensi sentimen IndoBERT"
    )
    parser.add_argument(
        "--input",
        default=str(PROCESSED_DATA_DIR / PREPROCESSED_REVIEWS_FILENAME),
        help="Path CSV preprocessed reviews",
    )
    parser.add_argument(
        "--model-dir",
        default=str(MODEL_OUTPUT_DIR / "baseline_indobert"),
        help="Direktori model IndoBERT yang sudah ditraining",
    )
    parser.add_argument(
        "--output-dir",
        default=str(LOG_DIR / "rating3_wordcloud"),
        help="Direktori output analisis",
    )
    parser.add_argument("--text-column", default="content_clean", help="Nama kolom teks")
    parser.add_argument("--score-column", default="score", help="Nama kolom rating")
    parser.add_argument("--date-column", default="at", help="Nama kolom tanggal review")
    parser.add_argument("--rating-value", type=int, default=3, help="Nilai rating target")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size inferensi")
    parser.add_argument("--max-length", type=int, default=128, help="Maksimal panjang token")
    parser.add_argument(
        "--min-words-for-wordcloud",
        type=int,
        default=5,
        help="Minimal jumlah kata unik agar wordcloud digenerate",
    )
    parser.add_argument("--max-words", type=int, default=120, help="Maksimal kata di wordcloud")
    parser.add_argument("--width", type=int, default=1600, help="Lebar gambar wordcloud")
    parser.add_argument("--height", type=int, default=900, help="Tinggi gambar wordcloud")
    return parser.parse_args()


def to_config(args: argparse.Namespace) -> Rating3WordcloudConfig:
    return Rating3WordcloudConfig(
        input_path=str(Path(args.input).resolve()),
        model_dir=str(Path(args.model_dir).resolve()),
        output_dir=str(Path(args.output_dir).resolve()),
        text_column=str(args.text_column),
        score_column=str(args.score_column),
        date_column=str(args.date_column),
        rating_value=int(args.rating_value),
        batch_size=max(1, int(args.batch_size)),
        max_length=max(8, int(args.max_length)),
        min_words_for_wordcloud=max(1, int(args.min_words_for_wordcloud)),
        max_words=max(10, int(args.max_words)),
        width=max(400, int(args.width)),
        height=max(300, int(args.height)),
    )


def _validate_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")


def load_rating_subset(config: Rating3WordcloudConfig) -> pd.DataFrame:
    input_path = Path(config.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"File input tidak ditemukan: {input_path}")

    df = pd.read_csv(input_path)
    _validate_columns(df, [config.text_column, config.score_column, config.date_column])

    work_df = df.copy()
    work_df[config.text_column] = work_df[config.text_column].fillna("").astype(str).str.strip()
    work_df = work_df[work_df[config.text_column] != ""]

    work_df[config.score_column] = pd.to_numeric(work_df[config.score_column], errors="coerce")
    work_df = work_df[work_df[config.score_column] == config.rating_value].copy()
    if work_df.empty:
        raise ValueError(f"Tidak ada data dengan rating={config.rating_value}")

    work_df[config.date_column] = pd.to_datetime(work_df[config.date_column], errors="coerce")
    work_df = work_df.dropna(subset=[config.date_column]).copy()
    if work_df.empty:
        raise ValueError("Semua tanggal invalid setelah parsing; tidak bisa membuat analisis per tahun")

    work_df["year"] = work_df[config.date_column].dt.year.astype(int)
    return work_df.reset_index(drop=True)


def add_predicted_sentiment(df: pd.DataFrame, config: Rating3WordcloudConfig) -> pd.DataFrame:
    model_dir = Path(config.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Direktori model tidak ditemukan: {model_dir}")

    predicted_labels, confidence_scores, _ = predict_sentiment(
        df[config.text_column].tolist(),
        model_path=model_dir,
        batch_size=config.batch_size,
        max_length=config.max_length,
    )

    result = df.copy()
    result["predicted_label"] = predicted_labels.astype(int)
    result["predicted_label_name"] = result["predicted_label"].map(LABEL_NAMES)
    result["confidence"] = confidence_scores.astype(float)
    return result


def build_wordcloud_image(
    texts: list[str],
    output_path: Path,
    title: str,
    *,
    width: int,
    height: int,
    max_words: int,
    min_words_for_wordcloud: int,
) -> dict[str, Any]:
    joined_text = " ".join(texts).strip()
    token_count = len(joined_text.split())
    unique_word_count = len(set(joined_text.split())) if joined_text else 0

    if token_count < min_words_for_wordcloud:
        return {
            "generated": False,
            "reason": f"Jumlah token terlalu sedikit ({token_count})",
            "tokenCount": int(token_count),
            "uniqueWordCount": int(unique_word_count),
            "path": str(output_path),
        }

    wc = WordCloud(
        width=width,
        height=height,
        background_color="white",
        colormap="viridis",
        max_words=max_words,
        collocations=False,
    ).generate(joined_text)

    ensure_parent_directory(output_path)
    fig = plt.figure(figsize=(16, 9))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "generated": True,
        "tokenCount": int(token_count),
        "uniqueWordCount": int(unique_word_count),
        "path": str(output_path),
    }


def _dominant_sentiment(counts: dict[str, int]) -> str:
    pos = int(counts.get("Positif", 0))
    neg = int(counts.get("Negatif", 0))
    neu = int(counts.get("Netral", 0))

    if pos > neg:
        return "cenderung Positif"
    if neg > pos:
        return "cenderung Negatif"
    if neu > 0:
        return "cenderung Netral (Positif vs Negatif seimbang)"
    return "tidak dapat ditentukan"


def summarize_sentiment(df_pred: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    overall_counts_series = df_pred["predicted_label_name"].value_counts()
    overall_counts = {label: int(overall_counts_series.get(label, 0)) for label in ["Negatif", "Netral", "Positif"]}
    overall_total = int(len(df_pred))
    overall_percent = {
        label: (count / overall_total * 100.0 if overall_total else 0.0)
        for label, count in overall_counts.items()
    }
    overall = {
        "totalReviews": overall_total,
        "counts": overall_counts,
        "percent": overall_percent,
        "conclusion": _dominant_sentiment(overall_counts),
    }

    by_year: list[dict[str, Any]] = []
    grouped = df_pred.groupby("year", dropna=False)
    for year, group in sorted(grouped, key=lambda item: item[0]):
        year_counts_series = group["predicted_label_name"].value_counts()
        year_counts = {label: int(year_counts_series.get(label, 0)) for label in ["Negatif", "Netral", "Positif"]}
        year_total = int(len(group))
        year_percent = {
            label: (count / year_total * 100.0 if year_total else 0.0)
            for label, count in year_counts.items()
        }

        by_year.append(
            {
                "year": int(year),
                "totalReviews": year_total,
                "counts": year_counts,
                "percent": year_percent,
                "conclusion": _dominant_sentiment(year_counts),
            }
        )

    return overall, by_year


def _format_pct(value: float) -> str:
    return f"{value:.2f}%"


def compute_word_frequency(texts: list[str]) -> pd.DataFrame:
    counter: Counter[str] = Counter()
    for text in texts:
        tokens = str(text).strip().split()
        counter.update(token for token in tokens if token)

    rows = [{"word": word, "count": int(count)} for word, count in counter.items()]
    freq_df = pd.DataFrame(rows)
    if freq_df.empty:
        return pd.DataFrame(columns=["word", "count"])

    freq_df = freq_df.sort_values(["count", "word"], ascending=[False, True]).reset_index(drop=True)
    return freq_df


def build_markdown_report(
    config: Rating3WordcloudConfig,
    generated_at: str,
    input_rows: int,
    filtered_rows: int,
    overall_summary: dict[str, Any],
    yearly_summary: list[dict[str, Any]],
    overall_wordcloud_meta: dict[str, Any],
    yearly_wordcloud_meta: list[dict[str, Any]],
    overall_freq_df: pd.DataFrame,
    yearly_freq_df: pd.DataFrame,
) -> str:
    lines: list[str] = [
        "# Analisis Wordcloud Rating 3",
        "",
        f"- Waktu proses: {generated_at}",
        f"- Input data: {config.input_path}",
        f"- Model inferensi: {config.model_dir}",
        f"- Total baris input: {input_rows}",
        f"- Total baris rating {config.rating_value} valid: {filtered_rows}",
        "",
        "## Kesimpulan Keseluruhan Rating 3",
        "",
        f"- Negatif: {overall_summary['counts']['Negatif']} ({_format_pct(overall_summary['percent']['Negatif'])})",
        f"- Netral: {overall_summary['counts']['Netral']} ({_format_pct(overall_summary['percent']['Netral'])})",
        f"- Positif: {overall_summary['counts']['Positif']} ({_format_pct(overall_summary['percent']['Positif'])})",
        f"- Kecenderungan: **{overall_summary['conclusion']}**",
        "",
        "## Kesimpulan Per Tahun (Rating 3)",
        "",
    ]

    for item in yearly_summary:
        lines.extend(
            [
                f"### Tahun {item['year']}",
                f"- Jumlah review: {item['totalReviews']}",
                f"- Negatif: {item['counts']['Negatif']} ({_format_pct(item['percent']['Negatif'])})",
                f"- Netral: {item['counts']['Netral']} ({_format_pct(item['percent']['Netral'])})",
                f"- Positif: {item['counts']['Positif']} ({_format_pct(item['percent']['Positif'])})",
                f"- Kecenderungan: **{item['conclusion']}**",
                "",
            ]
        )

    lines.extend(
        [
            "## Artefak Wordcloud",
            "",
            f"- Overall: `{overall_wordcloud_meta['path']}` | generated={overall_wordcloud_meta['generated']}",
        ]
    )

    for item in yearly_wordcloud_meta:
        lines.append(f"- Tahun {item['year']}: `{item['path']}` | generated={item['generated']}")

    overall_tidak_count = 0
    row_tidak_overall = overall_freq_df.loc[overall_freq_df["word"] == "tidak"]
    if not row_tidak_overall.empty:
        overall_tidak_count = int(row_tidak_overall["count"].iloc[0])

    lines.extend(
        [
            "",
            "## Frekuensi Kata (Teks)",
            "",
            f"- Frekuensi kata 'tidak' (overall): {overall_tidak_count}",
            "",
            "### Top 20 Kata Overall",
            "",
            "| Rank | Kata | Frekuensi |",
            "| --- | --- | --- |",
        ]
    )

    for rank, row in enumerate(overall_freq_df.head(20).itertuples(index=False), start=1):
        lines.append(f"| {rank} | {row.word} | {int(row.count)} |")

    if yearly_freq_df.empty:
        lines.extend(["", "Tidak ada data frekuensi kata per tahun."])
    else:
        for year in sorted(yearly_freq_df["year"].unique().tolist()):
            year_df = yearly_freq_df[yearly_freq_df["year"] == year].copy()
            year_df = year_df.sort_values(["count", "word"], ascending=[False, True]).reset_index(drop=True)

            tidak_count = 0
            row_tidak = year_df.loc[year_df["word"] == "tidak"]
            if not row_tidak.empty:
                tidak_count = int(row_tidak["count"].iloc[0])

            lines.extend(
                [
                    "",
                    f"### Top 20 Kata Tahun {int(year)}",
                    "",
                    f"- Frekuensi kata 'tidak' tahun {int(year)}: {tidak_count}",
                    "",
                    "| Rank | Kata | Frekuensi |",
                    "| --- | --- | --- |",
                ]
            )

            for rank, row in enumerate(year_df.head(20).itertuples(index=False), start=1):
                lines.append(f"| {rank} | {row.word} | {int(row.count)} |")

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    config = to_config(args)
    logger = get_logger("rating3_wordcloud", log_filename="rating3_wordcloud.log")
    ensure_base_directories()

    output_dir = Path(config.output_dir)
    images_dir = output_dir / "images"
    summary_json_path = output_dir / "summary.json"
    summary_md_path = output_dir / "summary.md"
    per_year_csv_path = output_dir / "sentiment_by_year.csv"
    overall_freq_csv_path = output_dir / "word_frequency_overall.csv"
    yearly_freq_csv_path = output_dir / "word_frequency_by_year.csv"

    try:
        raw_df = pd.read_csv(config.input_path)
        rating_df = load_rating_subset(config)
        predicted_df = add_predicted_sentiment(rating_df, config)

        overall_summary, yearly_summary = summarize_sentiment(predicted_df)

        overall_wordcloud_meta = build_wordcloud_image(
            predicted_df[config.text_column].tolist(),
            images_dir / f"rating_{config.rating_value}_overall.png",
            title=f"Wordcloud Rating {config.rating_value} - Keseluruhan",
            width=config.width,
            height=config.height,
            max_words=config.max_words,
            min_words_for_wordcloud=config.min_words_for_wordcloud,
        )

        yearly_wordcloud_meta: list[dict[str, Any]] = []
        for year, group in sorted(predicted_df.groupby("year"), key=lambda item: item[0]):
            meta = build_wordcloud_image(
                group[config.text_column].tolist(),
                images_dir / f"rating_{config.rating_value}_{int(year)}.png",
                title=f"Wordcloud Rating {config.rating_value} - Tahun {int(year)}",
                width=config.width,
                height=config.height,
                max_words=config.max_words,
                min_words_for_wordcloud=config.min_words_for_wordcloud,
            )
            meta["year"] = int(year)
            yearly_wordcloud_meta.append(meta)

        overall_freq_df = compute_word_frequency(predicted_df[config.text_column].tolist())

        yearly_freq_rows: list[dict[str, Any]] = []
        for year, group in sorted(predicted_df.groupby("year"), key=lambda item: item[0]):
            freq_df = compute_word_frequency(group[config.text_column].tolist())
            if freq_df.empty:
                continue

            freq_df = freq_df.copy()
            freq_df.insert(0, "year", int(year))
            yearly_freq_rows.extend(freq_df.to_dict(orient="records"))

        yearly_freq_df = pd.DataFrame(yearly_freq_rows)
        if yearly_freq_df.empty:
            yearly_freq_df = pd.DataFrame(columns=["year", "word", "count"])

        ensure_parent_directory(overall_freq_csv_path)
        overall_freq_df.to_csv(overall_freq_csv_path, index=False, encoding="utf-8")

        ensure_parent_directory(yearly_freq_csv_path)
        yearly_freq_df.to_csv(yearly_freq_csv_path, index=False, encoding="utf-8")

        generated_at = datetime.now().isoformat(timespec="seconds")
        summary = {
            "generatedAt": generated_at,
            "config": asdict(config),
            "inputRows": int(len(raw_df)),
            "filteredRows": int(len(rating_df)),
            "overallSentiment": overall_summary,
            "yearlySentiment": yearly_summary,
            "overallWordcloud": overall_wordcloud_meta,
            "yearlyWordcloud": yearly_wordcloud_meta,
            "wordFrequencyOverallPath": str(overall_freq_csv_path),
            "wordFrequencyByYearPath": str(yearly_freq_csv_path),
        }

        ensure_parent_directory(summary_json_path)
        summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        report = build_markdown_report(
            config=config,
            generated_at=generated_at,
            input_rows=int(len(raw_df)),
            filtered_rows=int(len(rating_df)),
            overall_summary=overall_summary,
            yearly_summary=yearly_summary,
            overall_wordcloud_meta=overall_wordcloud_meta,
            yearly_wordcloud_meta=yearly_wordcloud_meta,
            overall_freq_df=overall_freq_df,
            yearly_freq_df=yearly_freq_df,
        )
        ensure_parent_directory(summary_md_path)
        summary_md_path.write_text(report, encoding="utf-8")

        yearly_rows: list[dict[str, Any]] = []
        for item in yearly_summary:
            yearly_rows.append(
                {
                    "year": item["year"],
                    "total_reviews": item["totalReviews"],
                    "negative_count": item["counts"]["Negatif"],
                    "neutral_count": item["counts"]["Netral"],
                    "positive_count": item["counts"]["Positif"],
                    "negative_pct": round(float(item["percent"]["Negatif"]), 4),
                    "neutral_pct": round(float(item["percent"]["Netral"]), 4),
                    "positive_pct": round(float(item["percent"]["Positif"]), 4),
                    "conclusion": item["conclusion"],
                }
            )
        pd.DataFrame(yearly_rows).to_csv(per_year_csv_path, index=False, encoding="utf-8")

    except Exception as exc:
        logger.exception("Analisis wordcloud rating 3 gagal: %s", exc)
        return 1

    logger.info("Analisis wordcloud rating %s selesai", config.rating_value)
    logger.info("Summary JSON: %s", summary_json_path)
    logger.info("Summary MD: %s", summary_md_path)
    logger.info("CSV per tahun: %s", per_year_csv_path)
    logger.info("Word frequency overall CSV: %s", overall_freq_csv_path)
    logger.info("Word frequency by year CSV: %s", yearly_freq_csv_path)
    logger.info("Direktori gambar: %s", images_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
