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
# Winter precipitation-type composite
#
# This is a display product, not a second classifier.  It uses the phase
# already produced by the RAP/Modified-Bourgouin engine and colors each
# precipitation type with a compact intensity ramp similar to common
# radar precipitation-type displays:
#
#   Rain  -> yellow / orange / red
#   Snow  -> light blue / blue / deep blue
#   Ice   -> pink / magenta / purple
#   Mixed -> lavender / violet / purple
#
# Reflectivity controls the shade within each phase ramp, while the phase
# classification controls the hue family.
# ----------------------------------------------------------------------

TYPE_RAMP_LEVELS = np.arange(5.0, 100.0, 5.0, dtype=np.float32)

# Exact MRMS BR HiRes palette supplied for the standalone reflectivity
# product. Rain uses these exact breakpoints/colors so the composite rain
# field is visually identical to the normal MRMS reflectivity palette.
BR_HIRES_LEVELS = np.array(
    [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 34.5, 35.0, 40.0,
     45.0, 50.0, 57.5, 62.5, 67.5, 72.5, 77.5, 82.5, 95.0],
    dtype=np.float32,
)
BR_HIRES_COLORS = np.array(
    [(50, 50, 50), (14, 14, 90), (3, 79, 140), (7, 162, 182),
     (17, 229, 31), (12, 169, 20), (6, 104, 8), (255, 255, 0),
     (255, 194, 0), (255, 140, 0), (221, 0, 0), (107, 0, 0),
     (255, 163, 255), (238, 29, 244), (117, 0, 235),
     (0, 255, 219), (0, 76, 74), (0, 0, 0)],
    dtype=np.float32,
)

def _ramp_from_anchors(anchors):
    anchors = np.asarray(anchors, dtype=np.float32)
    return np.stack([
        np.interp(TYPE_RAMP_LEVELS, anchors[:, 0], anchors[:, 1]),
        np.interp(TYPE_RAMP_LEVELS, anchors[:, 0], anchors[:, 2]),
        np.interp(TYPE_RAMP_LEVELS, anchors[:, 0], anchors[:, 3]),
    ], axis=1)

TYPE_RAMP_COLORS = {
    # Rain: EXACT BR HiRes palette, sampled at the composite intensity levels.
    "rain": BR_HIRES_COLORS.copy(),
    # Other types use the same MRMS dBZ intensity scale but their own hue family.
    "snow": _ramp_from_anchors([
        (5, 115, 205, 235), (25, 65, 165, 238), (45, 25, 95, 225),
        (65, 10, 40, 185), (85, 35, 25, 130), (95, 120, 70, 190),
    ]),
    "ice": _ramp_from_anchors([
        (5, 245, 180, 215), (25, 232, 120, 195), (45, 210, 55, 170),
        (65, 160, 25, 140), (85, 100, 20, 110), (95, 130, 45, 155),
    ]),
    "mixed": _ramp_from_anchors([
        (5, 220, 190, 248), (25, 190, 135, 240), (45, 155, 85, 225),
        (65, 115, 50, 195), (85, 80, 30, 155), (95, 105, 45, 180),
    ]),
    "uncertain": _ramp_from_anchors([
        (5, 205, 208, 210), (25, 175, 180, 188), (45, 145, 150, 170),
        (65, 110, 115, 150), (85, 90, 95, 135), (95, 125, 130, 165),
    ]),
}


def _paint_intensity_ramp(
    rgba: np.ndarray,
    mask: np.ndarray,
    dbz: np.ndarray,
    colors: np.ndarray,
    alpha: int = 245,
) -> None:
    """Paint one phase mask with a reflectivity-dependent RGB ramp."""
    if not np.any(mask):
        return

    sample = np.clip(
        np.nan_to_num(dbz[mask], nan=TYPE_RAMP_LEVELS[0]),
        TYPE_RAMP_LEVELS[0],
        TYPE_RAMP_LEVELS[-1],
    )

    for channel in range(3):
        rgba[..., channel][mask] = np.rint(
            np.interp(sample, TYPE_RAMP_LEVELS, colors[:, channel])
        ).astype(np.uint8)

    rgba[..., 3][mask] = np.uint8(alpha)


def result_to_precip_type_rgba(
    result: ClassificationResult,
    reflectivity: np.ndarray,
    alpha: int = 245,
) -> np.ndarray:
    """
    Render the classified precipitation field as a screenshot-style
    precipitation-type composite.

    Important: this function does NOT determine precipitation type.  The
    existing phase engine supplies result.phase.  Reflectivity is used only
    to choose the shade/intensity inside each phase's color family.

    CLEAR pixels remain transparent.  Rain, snow, sleet, FZRA, and MIXED
    precipitation all receive visible colors.  UNKNOWN remains a neutral
    gray overlay so data limitations are visible without implying a phase.
    """
    dbz = np.asarray(reflectivity, dtype=np.float32)
    if dbz.shape != result.phase.shape:
        raise ValueError(
            f"Reflectivity shape {dbz.shape} does not match phase shape "
            f"{result.phase.shape}."
        )

    rgba = np.zeros((*result.phase.shape, 4), dtype=np.uint8)
    precip = np.isfinite(dbz) & (dbz >= 5.0)

    # Liquid rain.
    rain_mask = precip & (result.phase == RAIN)
    if np.any(rain_mask):
        sample = np.clip(np.nan_to_num(dbz[rain_mask], nan=BR_HIRES_LEVELS[0]), BR_HIRES_LEVELS[0], BR_HIRES_LEVELS[-1])
        idx = np.searchsorted(BR_HIRES_LEVELS, sample, side="right") - 1
        idx = np.clip(idx, 0, len(BR_HIRES_COLORS) - 1)
        rgba[..., :3][rain_mask] = BR_HIRES_COLORS[idx].astype(np.uint8)
        rgba[..., 3][rain_mask] = np.uint8(alpha)

    # Snow.
    _paint_intensity_ramp(
        rgba,
        precip & (result.phase == SNOW),
        dbz,
        TYPE_RAMP_COLORS["snow"],
        alpha=alpha,
    )

    # Sleet + freezing rain are intentionally grouped into an "Ice" hue
    # family for the visualization, while the scientific phase field remains
    # distinct underneath.
    ice_mask = precip & np.isin(result.phase, [SLEET, FZRA])
    _paint_intensity_ramp(
        rgba,
        ice_mask,
        dbz,
        TYPE_RAMP_COLORS["ice"],
        alpha=alpha,
    )

    # Mixed is explicitly distinct from the ice family.
    _paint_intensity_ramp(
        rgba,
        precip & (result.phase == MIXED),
        dbz,
        TYPE_RAMP_COLORS["mixed"],
        alpha=alpha,
    )

    # Unknown/uncertain precipitation gets a quieter gray.
    unknown = precip & (result.phase == UNKNOWN)
    rgba[unknown] = (145, 150, 155, min(alpha, 180))

    return rgba


# ----------------------------------------------------------------------
# MRMS reflectivity rendering
# ----------------------------------------------------------------------

def reflectivity_to_rgba(dbz: np.ndarray) -> np.ndarray:
    """Convert MRMS reflectivity using the exact BR HiRes palette.

    Native pixels are preserved; this function performs no spatial
    resampling. Values below 5 dBZ are transparent.
    """
    dbz = np.asarray(dbz, dtype=np.float32)
    rgba = np.zeros((*dbz.shape, 4), dtype=np.uint8)
    valid = np.isfinite(dbz) & (dbz >= 5.0)
    if not np.any(valid):
        return rgba
    sample = np.clip(dbz[valid], BR_HIRES_LEVELS[0], BR_HIRES_LEVELS[-1])
    idx = np.searchsorted(BR_HIRES_LEVELS, sample, side="right") - 1
    idx = np.clip(idx, 0, len(BR_HIRES_COLORS) - 1)
    rgba[..., :3][valid] = BR_HIRES_COLORS[idx].astype(np.uint8)
    rgba[..., 3][valid] = 255
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
        "grid_shape": [int(result.phase.shape[0]), int(result.phase.shape[1])],
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
