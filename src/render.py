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
from config import OUTPUT_DIR


# ----------------------------------------------------------------------
# Winter phase colors
#
# These are overlays only. Rain and clear remain transparent.
# ----------------------------------------------------------------------

PHASE_COLORS = {
    SNOW:   (55, 145, 255, 105),
    SLEET:  (185, 80, 220, 115),
    FZRA:   (225, 55, 70, 115),
    MIXED:  (220, 95, 195, 110),
    UNKNOWN: (145, 150, 155, 85),
}


# ----------------------------------------------------------------------
# MRMS reflectivity color table
#
# This provides the underlying radar image.
# ----------------------------------------------------------------------

def reflectivity_to_rgba(dbz: np.ndarray) -> np.ndarray:
    """
    Convert MRMS reflectivity into a radar-style RGBA image.

    Missing / very weak returns remain transparent.
    """

    h, w = dbz.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    valid = np.isfinite(dbz)

    # Very weak / no echo
    m = valid & (dbz < 5)
    rgba[m] = (0, 0, 0, 0)

    # 5–15 dBZ
    m = valid & (dbz >= 5) & (dbz < 15)
    rgba[m] = (80, 110, 125, 80)

    # 15–20 dBZ
    m = valid & (dbz >= 15) & (dbz < 20)
    rgba[m] = (80, 180, 110, 115)

    # 20–25 dBZ
    m = valid & (dbz >= 20) & (dbz < 25)
    rgba[m] = (100, 205, 100, 130)

    # 25–30 dBZ
    m = valid & (dbz >= 25) & (dbz < 30)
    rgba[m] = (175, 220, 70, 145)

    # 30–35 dBZ
    m = valid & (dbz >= 30) & (dbz < 35)
    rgba[m] = (240, 220, 60, 160)

    # 35–40 dBZ
    m = valid & (dbz >= 35) & (dbz < 40)
    rgba[m] = (245, 165, 45, 175)

    # 40–45 dBZ
    m = valid & (dbz >= 40) & (dbz < 45)
    rgba[m] = (240, 95, 45, 190)

    # 45–50 dBZ
    m = valid & (dbz >= 45) & (dbz < 50)
    rgba[m] = (225, 45, 45, 205)

    # 50–55 dBZ
    m = valid & (dbz >= 50) & (dbz < 55)
    rgba[m] = (210, 40, 100, 215)

    # 55+ dBZ
    m = valid & (dbz >= 55)
    rgba[m] = (180, 40, 180, 225)

    return rgba


# ----------------------------------------------------------------------
# Alpha compositing
# ----------------------------------------------------------------------

def alpha_composite(
    base: np.ndarray,
    overlay: np.ndarray,
) -> np.ndarray:
    """
    Alpha composite overlay over base.

    Both arrays are RGBA uint8.
    """

    base_f = base.astype(np.float32) / 255.0
    over_f = overlay.astype(np.float32) / 255.0

    oa = over_f[..., 3:4]
    ba = base_f[..., 3:4]

    out_a = oa + ba * (1.0 - oa)

    out_rgb = np.zeros_like(base_f[..., :3])

    valid = out_a[..., 0] > 0

    out_rgb[valid] = (
        over_f[..., :3][valid] * oa[valid]
        + base_f[..., :3][valid] * ba[valid] * (1.0 - oa[valid])
    ) / out_a[valid]

    out = np.zeros_like(base_f)
    out[..., :3] = out_rgb
    out[..., 3:4] = out_a

    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------
# Winter phase mask
# ----------------------------------------------------------------------

def result_to_phase_rgba(
    result: ClassificationResult,
) -> np.ndarray:

    h, w = result.phase.shape

    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    for phase, color in PHASE_COLORS.items():
        mask = result.phase == phase
        rgba[mask] = color

    # Explicitly keep rain and clear transparent.
    rgba[result.phase == RAIN] = (0, 0, 0, 0)
    rgba[result.phase == CLEAR] = (0, 0, 0, 0)

    return rgba


# ----------------------------------------------------------------------
# Main radar + winter mask product
# ----------------------------------------------------------------------

def result_to_rgba(
    result: ClassificationResult,
    reflectivity: np.ndarray | None = None,
) -> np.ndarray:
    """
    Create the final displayed product.

    If reflectivity is supplied:
        MRMS reflectivity is rendered underneath
        and winter precipitation is overlaid.

    If reflectivity is not supplied:
        returns the winter phase mask only.
    """

    if reflectivity is None:
        return result_to_phase_rgba(result)

    radar = reflectivity_to_rgba(reflectivity)
    phase = result_to_phase_rgba(result)

    return alpha_composite(radar, phase)


# ----------------------------------------------------------------------
# Save PNG
# ----------------------------------------------------------------------

def save_rgba_png(
    rgba: np.ndarray,
    path: Path,
) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)

    Image.fromarray(
        rgba,
        mode="RGBA",
    ).save(
        path,
        optimize=True,
    )


# ----------------------------------------------------------------------
# Metadata
# ----------------------------------------------------------------------

def write_metadata(
    result: ClassificationResult,
    path: Path,
) -> None:

    unique, counts = np.unique(
        result.phase,
        return_counts=True,
    )

    counts_by_phase = {
        str(int(k)): int(v)
        for k, v in zip(unique, counts)
    }

    phase_names = {
        CLEAR: "clear",
        RAIN: "rain",
        SNOW: "snow",
        SLEET: "sleet",
        FZRA: "freezing_rain",
        MIXED: "mixed",
        UNKNOWN: "uncertain",
    }

    named_counts = {
        phase_names.get(int(k), str(int(k))): int(v)
        for k, v in zip(unique, counts)
    }

    winter_pixels = int(
        np.count_nonzero(
            np.isin(
                result.phase,
                [SNOW, SLEET, FZRA, MIXED, UNKNOWN],
            )
        )
    )

    rain_pixels = int(
        np.count_nonzero(result.phase == RAIN)
    )

    clear_pixels = int(
        np.count_nonzero(result.phase == CLEAR)
    )

    metadata = {
        "counts_by_phase": counts_by_phase,
        "counts_named": named_counts,
        "total_pixels": int(result.phase.size),
        "winter_or_uncertain_pixels": winter_pixels,
        "rain_pixels": rain_pixels,
        "clear_pixels": clear_pixels,
    }

    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
