from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from classifier import CLEAR, FZRA, MIXED, RAIN, SLEET, SNOW, UNKNOWN, ClassificationResult
from config import OUTPUT_DIR

# RGBA color table. Colors intentionally distinguish phase from intensity.
RAIN_COLORS = {
    0: (0, 0, 0, 0),
    1: (70, 200, 90, 180),
    2: (30, 170, 70, 195),
    3: (245, 220, 50, 205),
    4: (245, 135, 35, 215),
    5: (220, 40, 40, 225),
}
PHASE_COLORS = {
    SNOW: (55, 145, 255, 215),
    SLEET: (235, 105, 210, 220),
    FZRA: (215, 45, 55, 225),
    MIXED: (238, 125, 205, 215),
}


def result_to_rgba(result: ClassificationResult) -> np.ndarray:
    h, w = result.phase.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    for intensity, color in RAIN_COLORS.items():
        mask = (result.phase == RAIN) & (result.intensity == intensity)
        rgba[mask] = color

    for phase, color in PHASE_COLORS.items():
        # Use intensity to modestly vary winter color opacity.
        mask = result.phase == phase
        rgba[mask] = color

    unknown = result.phase == UNKNOWN
    rgba[unknown] = (120, 120, 120, 100)
    return rgba


def save_rgba_png(rgba: np.ndarray, path: Path) -> None:
    Image.fromarray(rgba, mode="RGBA").save(path, optimize=True)


def write_metadata(result: ClassificationResult, path: Path) -> None:
    unique, counts = np.unique(result.phase, return_counts=True)
    counts_by_phase = {str(int(k)): int(v) for k, v in zip(unique, counts)}
    metadata = {"counts_by_phase": counts_by_phase}
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
