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


# ----------------------------------------------------------------------
# Winter phase colors
#
# The winter mask is intentionally translucent so the radar beneath
# remains visible on the interactive map.
# ----------------------------------------------------------------------

PHASE_COLORS = {
    SNOW:    (55, 145, 255, 105),
    SLEET:   (185, 80, 220, 115),
    FZRA:    (225, 55, 70, 115),
    MIXED:   (220, 95, 195, 110),
    UNKNOWN: (145, 150, 155, 85),
}


# ----------------------------------------------------------------------
# MRMS reflectivity rendering
# ----------------------------------------------------------------------

def reflectivity_to_rgba(dbz: np.ndarray) -> np.ndarray:
    """
    Convert MRMS reflectivity to a transparent radar-style RGBA PNG.

    Weak/no echo is transparent so the geographic basemap remains
    visible underneath the radar.
    """

    rgba = np.zeros(
        (*dbz.shape, 4),
        dtype=np.uint8,
    )

    valid = np.isfinite(dbz)

    m = valid & (dbz < 5)
    rgba[m] = (0, 0, 0, 0)

    m = valid & (dbz >= 5) & (dbz < 15)
    rgba[m] = (90, 125, 140, 90)

    m = valid & (dbz >= 15) & (dbz < 20)
    rgba[m] = (80, 180, 110, 115)

    m = valid & (dbz >= 20) & (dbz < 25)
    rgba[m] = (105, 205, 105, 130)

    m = valid & (dbz >= 25) & (dbz < 30)
    rgba[m] = (180, 225, 70, 145)

    m = valid & (dbz >= 30) & (dbz < 35)
    rgba[m] = (242, 220, 60, 160)

    m = valid & (dbz >= 35) & (dbz < 40)
    rgba[m] = (246, 170, 45, 175)

    m = valid & (dbz >= 40) & (dbz < 45)
    rgba[m] = (242, 100, 42, 190)

    m = valid & (dbz >= 45) & (dbz < 50)
    rgba[m] = (228, 48, 45, 205)

    m = valid & (dbz >= 50) & (dbz < 55)
    rgba[m] = (210, 40, 105, 215)

    m = valid & (dbz >= 55)
    rgba[m] = (180, 45, 180, 225)

    return rgba


# ----------------------------------------------------------------------
# Winter phase mask rendering
# ----------------------------------------------------------------------

def result_to_phase_rgba(
    result: ClassificationResult,
) -> np.ndarray:
    """
    Create a standalone winter phase RGBA overlay.

    Rain and clear pixels are explicitly transparent.
    """

    rgba = np.zeros(
        (*result.phase.shape, 4),
        dtype=np.uint8,
    )

    for phase, color in PHASE_COLORS.items():
        mask = result.phase == phase
        rgba[mask] = color

    rgba[result.phase == RAIN] = (0, 0, 0, 0)
    rgba[result.phase == CLEAR] = (0, 0, 0, 0)

    return rgba


# ----------------------------------------------------------------------
# Optional combined product
# ----------------------------------------------------------------------

def result_to_rgba(
    result: ClassificationResult,
    reflectivity: np.ndarray | None = None,
) -> np.ndarray:
    """
    Return either the pure phase mask or radar + phase overlay.

    The interactive viewer uses the two layers separately, but this
    combined function is retained for research/export products.
    """

    phase_rgba = result_to_phase_rgba(result)

    if reflectivity is None:
        return phase_rgba

    radar = reflectivity_to_rgba(reflectivity)

    return alpha_composite(
        radar,
        phase_rgba,
    )


# ----------------------------------------------------------------------
# Alpha compositing
# ----------------------------------------------------------------------

def alpha_composite(
    base: np.ndarray,
    overlay: np.ndarray,
) -> np.ndarray:

    base_f = base.astype(np.float32) / 255.0
    over_f = overlay.astype(np.float32) / 255.0

    oa = over_f[..., 3:4]
    ba = base_f[..., 3:4]

    out_a = oa + ba * (1.0 - oa)

    out_rgb = np.zeros_like(
        base_f[..., :3]
    )

    valid = out_a[..., 0] > 0

    if np.any(valid):
        out_rgb[valid] = (
            over_f[..., :3][valid] * oa[valid]
            + base_f[..., :3][valid]
            * ba[valid]
            * (1.0 - oa[valid])
        ) / out_a[valid]

    out = np.zeros_like(base_f)

    out[..., :3] = out_rgb
    out[..., 3:4] = out_a

    return np.clip(
        out * 255.0,
        0,
        255,
    ).astype(np.uint8)


# ----------------------------------------------------------------------
# PNG writer
# ----------------------------------------------------------------------

def save_rgba_png(
    rgba: np.ndarray,
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
        phase_names.get(
            int(k),
            str(int(k)),
        ): int(v)
        for k, v in zip(unique, counts)
    }

    winter_pixels = int(
        np.count_nonzero(
            np.isin(
                result.phase,
                [
                    SNOW,
                    SLEET,
                    FZRA,
                    MIXED,
                    UNKNOWN,
                ],
            )
        )
    )

    rain_pixels = int(
        np.count_nonzero(
            result.phase == RAIN
        )
    )

    clear_pixels = int(
        np.count_nonzero(
            result.phase == CLEAR
        )
    )

    metadata = {
        "counts_by_phase": counts_by_phase,
        "counts_named": named_counts,
        "total_pixels": int(result.phase.size),
        "winter_or_uncertain_pixels": winter_pixels,
        "rain_pixels": rain_pixels,
        "clear_pixels": clear_pixels,
    }

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )
