from __future__ import annotations

from datetime import datetime
from pathlib import Path

from src.config import STAGE_OUTPUT_FILES


def get_output_path(stage: str, with_timestamp: bool = False) -> Path:
    """Menghasilkan path output standar berdasarkan nama stage proses."""
    if stage not in STAGE_OUTPUT_FILES:
        raise ValueError(f"Stage tidak dikenali: {stage}")

    base_path = STAGE_OUTPUT_FILES[stage]
    if not with_timestamp:
        return base_path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_path.with_name(f"{base_path.stem}_{timestamp}{base_path.suffix}")