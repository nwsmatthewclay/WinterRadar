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


# Current MRMS PrecipFlag definitions.
PRECIPFLAG_NO_PRECIP = 0
PRECIPFLAG_WARM_STRATIFORM = 1
PRECIPFLAG_SNOW = 3
PRECIPFLAG_CONVECTION = 6
PRECIPFLAG_HAIL = 7
PRECIPFLAG_COOL_STRATIFORM = 10
PRECIPFLAG_TROPICAL_STRATIFORM = 91
PRECIPFLAG_TROPICAL_CONVECTION = 96


def rain_intensity_dbz(dbz: np.ndarray) -> np.ndarray:
    """Rain intensity class: 0 none, 1-5 increasing."""
    out = np.zeros(dbz.shape, dtype=np.uint8)

    valid = np.isfinite(dbz)

    out[valid & (dbz >= 15) & (dbz < 25)] = 1
    out[valid & (dbz >= 25) & (dbz < 35)] = 2
    out[valid & (dbz >= 35) & (dbz < 45)] = 3
    out[valid & (dbz >= 45) & (dbz < 55)] = 4
    out[valid & (dbz >= 55)] = 5

    return out


def classify_initial(
    reflectivity: np.ndarray,
    precip_flag: np.ndarray,
    bb_top_m: np.ndarray,
    bb_bottom_m: np.ndarray,
    wetbulb_c: np.ndarray,
    freezing_level_m: np.ndarray,
    rqi: np.ndarray,
) -> ClassificationResult:
    """
    Clean first-pass radar-assisted classifier.

    This does NOT yet distinguish sleet from freezing rain.
    That requires the vertical thermodynamic profile that we will
    add in the next science stage.
    """
    phase = np.full(
        reflectivity.shape,
        CLEAR,
        dtype=np.uint8,
    )

    confidence = np.zeros(
        reflectivity.shape,
        dtype=np.float32,
    )

    valid_ref = np.isfinite(reflectivity)

    precip = (
        valid_ref
        & (reflectivity >= 10.0)
    )

    phase[precip] = RAIN

    valid_wb = np.isfinite(wetbulb_c)

    cold_surface = (
        valid_wb
        & (wetbulb_c <= 0.0)
    )

    warm_surface = (
        valid_wb
        & (wetbulb_c > 0.5)
    )

    # MRMS flag 3 = snow.
    snow_flag = (
        np.isfinite(precip_flag)
        & (
            precip_flag.astype(np.int16)
            == PRECIPFLAG_SNOW
        )
    )

    has_bb = (
        np.isfinite(bb_top_m)
        & np.isfinite(bb_bottom_m)
        & (bb_top_m >= 0.0)
        & (bb_bottom_m >= 0.0)
        & (bb_top_m >= bb_bottom_m)
    )

    # Cold surface + explicit snow flag OR no detected melting layer.
    snow = (
        precip
        & cold_surface
        & (snow_flag | ~has_bb)
    )

    phase[snow] = SNOW
    confidence[snow] = 0.72

    # Cold surface + melting layer = mixed for now.
    mixed = (
        precip
        & cold_surface
        & has_bb
    )

    phase[mixed] = MIXED
    confidence[mixed] = 0.55

    # Warm surface = rain.
    rain = (
        precip
        & warm_surface
    )

    phase[rain] = RAIN
    confidence[rain] = 0.70

    # Reduce confidence when radar quality is poor.
    good_rqi = np.clip(
        np.nan_to_num(rqi, nan=0.0),
        0.0,
        1.0,
    )

    confidence *= (
        0.5
        + 0.5 * good_rqi
    )

    intensity = rain_intensity_dbz(
        reflectivity
    )

    # Winter precip isn't using the rain intensity scale yet.
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
