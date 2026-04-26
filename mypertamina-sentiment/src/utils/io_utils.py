from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


def ensure_directory(path: Path | str) -> Path:
    """Memastikan satu direktori tersedia lalu mengembalikan Path-nya."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_parent_directory(file_path: Path | str) -> Path:
    """Memastikan parent directory dari file target tersedia."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_csv_records(
    records: Iterable[dict],
    output_path: Path | str,
    fieldnames: list[str] | None = None,
) -> Path:
    """Menyimpan iterable of dict menjadi CSV dengan header otomatis."""
    rows = list(records)
    output = ensure_parent_directory(output_path)

    if not rows and fieldnames is None:
        raise ValueError("records kosong dan fieldnames tidak diberikan")

    if fieldnames is None:
        fieldnames = list(rows[0].keys())

    with output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output


def load_csv_records(input_path: Path | str) -> list[dict[str, str]]:
    """Membaca CSV menjadi list of dict untuk pemrosesan sederhana."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        return list(reader)