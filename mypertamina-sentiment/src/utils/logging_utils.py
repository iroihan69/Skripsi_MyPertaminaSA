from __future__ import annotations

import logging
from pathlib import Path

from src.config import LOG_DIR


def get_logger(name: str, log_filename: str | None = None) -> logging.Logger:
    """Membuat logger standar proyek: console + optional file handler."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if getattr(logger, "_is_project_logger", False):
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_filename:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_path = Path(LOG_DIR) / log_filename
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger._is_project_logger = True
    return logger