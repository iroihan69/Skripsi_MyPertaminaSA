from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud


STOPWORDS_ID = {
    "dan",
    "yang",
    "di",
    "ke",
    "untuk",
    "dari",
    "dengan",
    "adalah",
    "atau",
    "ini",
    "itu",
    "pada",
    "telah",
    "dapat",
    "juga",
    "akan",
    "seperti",
    "oleh",
    "sudah",
    "kalau",
    "karena",
    "menjadi",
    "saat",
    "tapi",
    "namun",
    "sesuai",
    "tanpa",
    "kali",
    "sering",
    "hanya",
    "lebih",
    "dalam",
    "nya",
    "an",
    "a",
    "i",
    "u",
    "e",
    "o",
    "ya",
    "iya",
    "lah",
    "nih",
    "deh",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate wordcloud PNG untuk RM2 (overall, per tahun, dan per sentimen per tahun)."
    )
    parser.add_argument(
        "--input",
        default="data/predictions/predictions_revisi_2kelas_20260421.csv",
        help="Path CSV prediksi.",
    )
    parser.add_argument(
        "--output-dir",
        default="logs/wordcloud_rm2_2022_2025",
        help="Folder output PNG.",
    )
    parser.add_argument(
        "--text-column",
        default="content_clean",
        help="Nama kolom teks bersih.",
    )
    parser.add_argument("--date-column", default="at", help="Nama kolom tanggal.")
    parser.add_argument(
        "--label-column",
        default="predicted_label",
        help="Nama kolom label prediksi (0=Negatif, 1=Positif).",
    )
    parser.add_argument("--start-year", type=int, default=2022, help="Tahun awal.")
    parser.add_argument("--end-year", type=int, default=2025, help="Tahun akhir.")
    parser.add_argument("--max-words", type=int, default=120, help="Maksimum kata wordcloud.")
    parser.add_argument("--width", type=int, default=1600, help="Lebar gambar.")
    parser.add_argument("--height", type=int, default=900, help="Tinggi gambar.")
    return parser.parse_args()


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"\b[a-z]+\b", str(text).lower())
    return [w for w in tokens if len(w) > 2 and w not in STOPWORDS_ID]


def _build_text(series: pd.Series) -> str:
    all_tokens: list[str] = []
    for value in series.fillna("").astype(str):
        all_tokens.extend(_tokenize(value))
    return " ".join(all_tokens)


def _save_wordcloud(text_blob: str, out_path: Path, title: str, max_words: int, width: int, height: int) -> bool:
    if not text_blob.strip():
        return False

    wc = WordCloud(
        width=width,
        height=height,
        background_color="white",
        colormap="viridis",
        max_words=max_words,
        random_state=42,
        collocations=False,
    ).generate(text_blob)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(16, 9))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input tidak ditemukan: {input_path}")

    df = pd.read_csv(input_path)
    required = [args.text_column, args.date_column, args.label_column]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")

    work = df.copy()
    work[args.date_column] = pd.to_datetime(work[args.date_column], errors="coerce", utc=True)
    work = work.dropna(subset=[args.date_column])
    work["year"] = work[args.date_column].dt.year.astype(int)
    work = work[work["year"].between(args.start_year, args.end_year)]
    work = work.dropna(subset=[args.text_column]).copy()

    if work.empty:
        raise ValueError("Data kosong setelah filter tahun.")

    # Overall all sentiment
    overall_blob = _build_text(work[args.text_column])
    _save_wordcloud(
        overall_blob,
        output_dir / "overall_2022_2025.png",
        "Wordcloud Keseluruhan 2022-2025",
        args.max_words,
        args.width,
        args.height,
    )

    # Overall by sentiment
    sentiment_map = {0: "negatif", 1: "positif"}
    for label_id, label_name in sentiment_map.items():
        subset = work[work[args.label_column].astype(int) == label_id]
        text_blob = _build_text(subset[args.text_column])
        _save_wordcloud(
            text_blob,
            output_dir / f"overall_{label_name}_2022_2025.png",
            f"Wordcloud {label_name.title()} 2022-2025",
            args.max_words,
            args.width,
            args.height,
        )

    # Per year and per year x sentiment
    for year in range(args.start_year, args.end_year + 1):
        yearly = work[work["year"] == year]
        if yearly.empty:
            continue

        yearly_blob = _build_text(yearly[args.text_column])
        _save_wordcloud(
            yearly_blob,
            output_dir / f"year_{year}_overall.png",
            f"Wordcloud Keseluruhan Tahun {year}",
            args.max_words,
            args.width,
            args.height,
        )

        for label_id, label_name in sentiment_map.items():
            year_sent = yearly[yearly[args.label_column].astype(int) == label_id]
            sent_blob = _build_text(year_sent[args.text_column])
            _save_wordcloud(
                sent_blob,
                output_dir / f"year_{year}_{label_name}.png",
                f"Wordcloud {label_name.title()} Tahun {year}",
                args.max_words,
                args.width,
                args.height,
            )

    print(f"Wordcloud selesai digenerate di: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
