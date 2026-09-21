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
    """Convert MRMS reflectivity to RGBA using the supplied BR HiRes palette.

    Palette breakpoints come directly from BR HiRes.pal. Colors are linearly
    interpolated between breakpoints. Values below 5 dBZ remain transparent
    so the basemap shows through in non-precipitating areas. Reflectivity
    pixels are fully opaque so 0% UI transparency is truly full-strength radar.
    """
    dbz = np.asarray(dbz, dtype=np.float32)
    rgba = np.zeros((*dbz.shape, 4), dtype=np.uint8)
    valid = np.isfinite(dbz) & (dbz >= 5.0)

    if not np.any(valid):
        return rgba

    levels = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 34.5, 35.0, 40.0, 45.0, 50.0, 57.5, 62.5, 67.5, 72.5, 77.5, 82.5, 95.0], dtype=np.float32)
    colors = np.array([(50, 50, 50), (14, 14, 90), (3, 79, 140), (7, 162, 182), (17, 229, 31), (12, 169, 20), (6, 104, 8), (255, 255, 0), (255, 194, 0), (255, 140, 0), (221, 0, 0), (107, 0, 0), (255, 163, 255), (238, 29, 244), (117, 0, 235), (0, 255, 219), (0, 76, 74), (0, 0, 0)], dtype=np.float32)
    sample = np.clip(dbz[valid], levels[0], levels[-1])

    for channel in range(3):
        rgba[..., channel][valid] = np.rint(
            np.interp(sample, levels, colors[:, channel])
        ).astype(np.uint8)

    rgba[..., 3][valid] = 255
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
