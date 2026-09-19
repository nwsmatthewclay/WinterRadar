from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ClassificationResult:
    phase: np.ndarray
    confidence: np.ndarray
    intensity: np.ndarray


CLEAR = 0
RAIN = 1
SNOW = 2
SLEET = 3
FZRA = 4
MIXED = 5
UNKNOWN = 9

# MRMS PrecipFlag is a broad surface-precipitation context field.  The NOAA
# operational table describes it as convective/stratiform/tropical/hail/snow
# context rather than a complete precipitation-type diagnosis.
PRECIPFLAG_NO_PRECIP = 0
PRECIPFLAG_WARM_STRATIFORM = 1
PRECIPFLAG_SNOW = 3
PRECIPFLAG_CONVECTION = 6
PRECIPFLAG_HAIL = 7
PRECIPFLAG_COOL_STRATIFORM = 10
PRECIPFLAG_TROPICAL_STRATIFORM = 91
PRECIPFLAG_TROPICAL_CONVECTION = 96


def rain_intensity_dbz(dbz: np.ndarray) -> np.ndarray:
    out = np.zeros(dbz.shape, dtype=np.uint8)
    valid = np.isfinite(dbz)
    out[valid & (dbz >= 15) & (dbz < 25)] = 1
    out[valid & (dbz >= 25) & (dbz < 35)] = 2
    out[valid & (dbz >= 35) & (dbz < 45)] = 3
    out[valid & (dbz >= 45) & (dbz < 55)] = 4
    out[valid & (dbz >= 55)] = 5
    return out


def _same_shape_or_default(
    arr: np.ndarray | None,
    shape: tuple[int, int],
    default: float,
) -> np.ndarray:
    if arr is None:
        return np.full(shape, default, dtype=np.float32)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape != shape:
        raise ValueError(f"Grid-shape mismatch: expected {shape}, got {arr.shape}")
    return arr


def classify_initial(
    reflectivity: np.ndarray,
    precip_flag: np.ndarray | None = None,
    bb_top_m: np.ndarray | None = None,
    bb_bottom_m: np.ndarray | None = None,
    wetbulb_c: np.ndarray | None = None,
    freezing_level_m: np.ndarray | None = None,
    rqi: np.ndarray | None = None,
) -> ClassificationResult:
    """
    Conservative first-pass winter precipitation classifier.

    The live product is designed to remain useful even when optional
    phase-support fields are temporarily unavailable. Missing optional fields
    result in lower-confidence/unknown phase rather than aborting the radar
    pipeline.
    """
    del freezing_level_m  # Reserved for the later vertical-profile engine.

    reflectivity = np.asarray(reflectivity, dtype=np.float32)
    shape = reflectivity.shape
    if reflectivity.ndim != 2:
        raise ValueError(f"Reflectivity must be 2-D, got {shape}")

    precip_flag = _same_shape_or_default(precip_flag, shape, np.nan)
    bb_top_m = _same_shape_or_default(bb_top_m, shape, np.nan)
    bb_bottom_m = _same_shape_or_default(bb_bottom_m, shape, np.nan)
    wetbulb_c = _same_shape_or_default(wetbulb_c, shape, np.nan)
    rqi = _same_shape_or_default(rqi, shape, 1.0)

    phase = np.full(shape, CLEAR, dtype=np.uint8)
    confidence = np.zeros(shape, dtype=np.float32)

    precip = np.isfinite(reflectivity) & (reflectivity >= 10.0)
    phase[precip] = RAIN

    valid_wb = np.isfinite(wetbulb_c) & (wetbulb_c > -90.0) & (wetbulb_c < 90.0)
    cold_surface = valid_wb & (wetbulb_c <= 0.0)
    warm_surface = valid_wb & (wetbulb_c > 0.5)
    near_freezing = valid_wb & (wetbulb_c > 0.0) & (wetbulb_c <= 0.5)

    valid_flag = np.isfinite(precip_flag)
    flag_int = np.zeros(shape, dtype=np.int16)
    flag_int[valid_flag] = precip_flag[valid_flag].astype(np.int16)
    snow_flag = valid_flag & (flag_int == PRECIPFLAG_SNOW)

    has_bb = (
        np.isfinite(bb_top_m)
        & np.isfinite(bb_bottom_m)
        & (bb_top_m >= 0.0)
        & (bb_bottom_m >= 0.0)
        & (bb_top_m >= bb_bottom_m)
    )

    # Clear snow signal first.
    snow = precip & cold_surface & (snow_flag | ~has_bb)
    phase[snow] = SNOW
    confidence[snow] = np.where(snow_flag[snow], 0.76, 0.64)

    # Cold surface + evidence of a melt layer remains mixed until a proper
    # vertical thermal-profile algorithm is applied.
    mixed = precip & cold_surface & has_bb
    phase[mixed] = MIXED
    confidence[mixed] = 0.55

    # Warm surface remains rain.
    rain = precip & warm_surface
    phase[rain] = RAIN
    confidence[rain] = 0.70

    # 0.0 to +0.5 C is intentionally treated as uncertain/mixed rather than
    # forcing a surface phase answer.
    near_surface = precip & near_freezing
    phase[near_surface] = MIXED
    confidence[near_surface] = 0.45

    # If the wet-bulb field is unavailable, preserve a snow flag when present;
    # otherwise mark precipitation as uncertain rather than falsely declaring
    # rain or snow.
    missing_thermal = precip & ~valid_wb
    snow_without_thermal = missing_thermal & snow_flag
    phase[snow_without_thermal] = SNOW
    confidence[snow_without_thermal] = 0.50

    uncertain = missing_thermal & ~snow_flag
    phase[uncertain] = UNKNOWN
    confidence[uncertain] = 0.20

    good_rqi = np.clip(np.nan_to_num(rqi, nan=1.0), 0.0, 1.0)
    confidence *= 0.5 + 0.5 * good_rqi

    intensity = rain_intensity_dbz(reflectivity)
    intensity[phase != RAIN] = 0

    bad = ~precip
    phase[bad] = CLEAR
    confidence[bad] = 0.0
    intensity[bad] = 0

    return ClassificationResult(
        phase=phase,
        confidence=confidence,
        intensity=intensity,
    )
