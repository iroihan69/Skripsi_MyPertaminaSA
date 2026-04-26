from __future__ import annotations

from collections.abc import Iterable


def validate_required_columns(headers: Iterable[str], required_columns: list[str]) -> list[str]:
    """Mengembalikan daftar kolom wajib yang belum tersedia."""
    available = {column.strip() for column in headers}
    return [column for column in required_columns if column not in available]


def validate_non_empty_text(value: str | None) -> bool:
    """Validasi teks tidak null dan tidak kosong setelah trim."""
    return bool(value and value.strip())