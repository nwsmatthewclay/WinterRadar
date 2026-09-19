from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from classifier import (
    CLEAR,
    FZRA,
    MIXED,
    RAIN,
    SLEET,
    SNOW,
    UNKNOWN,
    ClassificationResult,
)

PHASE_COLORS = {
    SNOW: (55, 145, 255, 105),
    SLEET: (185, 80, 220, 115),
    FZRA: (225, 55, 70, 115),
    MIXED: (220, 95, 195, 110),
    UNKNOWN: (145, 150, 155, 85),
}


def reflectivity_to_rgba(dbz: np.ndarray) -> np.ndarray:
    dbz = np.asarray(dbz, dtype=np.float32)
    rgba = np.zeros((*dbz.shape, 4), dtype=np.uint8)
    valid = np.isfinite(dbz)

    bands = [
        (5, 15, (90, 125, 140, 90)),
        (15, 20, (80, 180, 110, 115)),
        (20, 25, (105, 205, 105, 130)),
        (25, 30, (180, 225, 70, 145)),
        (30, 35, (242, 220, 60, 160)),
        (35, 40, (246, 170, 45, 175)),
        (40, 45, (242, 100, 42, 190)),
        (45, 50, (228, 48, 45, 205)),
        (50, 55, (210, 40, 105, 215)),
    ]

    for low, high, color in bands:
        mask = valid & (dbz >= low) & (dbz < high)
        rgba[mask] = color

    rgba[valid & (dbz >= 55)] = (180, 45, 180, 225)
    return rgba


def result_to_phase_rgba(result: ClassificationResult) -> np.ndarray:
    rgba = np.zeros((*result.phase.shape, 4), dtype=np.uint8)
    for phase, color in PHASE_COLORS.items():
        rgba[result.phase == phase] = color
    rgba[result.phase == RAIN] = (0, 0, 0, 0)
    rgba[result.phase == CLEAR] = (0, 0, 0, 0)
    return rgba


def result_to_rgba(
    result: ClassificationResult,
    reflectivity: np.ndarray | None = None,
) -> np.ndarray:
    phase_rgba = result_to_phase_rgba(result)
    if reflectivity is None:
        return phase_rgba
    return alpha_composite(reflectivity_to_rgba(reflectivity), phase_rgba)


def alpha_composite(base: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    base_f = base.astype(np.float32) / 255.0
    over_f = overlay.astype(np.float32) / 255.0
    oa = over_f[..., 3:4]
    ba = base_f[..., 3:4]
    out_a = oa + ba * (1.0 - oa)

    out_rgb = np.zeros_like(base_f[..., :3])
    valid = out_a[..., 0] > 0
    if np.any(valid):
        out_rgb[valid] = (
            over_f[..., :3][valid] * oa[valid]
            + base_f[..., :3][valid] * ba[valid] * (1.0 - oa[valid])
        ) / out_a[valid]

    out = np.zeros_like(base_f)
    out[..., :3] = out_rgb
    out[..., 3:4] = out_a
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def save_rgba_png(rgba: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(rgba, dtype=np.uint8), mode="RGBA").save(
        path,
        optimize=True,
    )


def write_metadata(result: ClassificationResult, path: Path) -> None:
    unique, counts = np.unique(result.phase, return_counts=True)

    phase_names = {
        CLEAR: "clear",
        RAIN: "rain",
        SNOW: "snow",
        SLEET: "sleet",
        FZRA: "freezing_rain",
        MIXED: "mixed",
        UNKNOWN: "uncertain",
    }

    counts_by_phase = {str(int(k)): int(v) for k, v in zip(unique, counts)}
    named_counts = {
        phase_names.get(int(k), str(int(k))): int(v)
        for k, v in zip(unique, counts)
    }

    metadata = {
        "counts_by_phase": counts_by_phase,
        "counts_named": named_counts,
        "total_pixels": int(result.phase.size),
        "winter_or_uncertain_pixels": int(
            np.count_nonzero(np.isin(result.phase, [SNOW, SLEET, FZRA, MIXED, UNKNOWN]))
        ),
        "rain_pixels": int(np.count_nonzero(result.phase == RAIN)),
        "clear_pixels": int(np.count_nonzero(result.phase == CLEAR)),
        "grid_shape": [int(result.phase.shape[0]), int(result.phase.shape[1])],
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
