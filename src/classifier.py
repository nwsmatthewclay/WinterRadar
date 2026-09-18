from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# ---------------------------------------------------------------------
# Phase codes
# ---------------------------------------------------------------------

CLEAR = 0
RAIN = 1
SNOW = 2
SLEET = 3
FZRA = 4
MIXED = 5
UNKNOWN = 9


# ---------------------------------------------------------------------
# MRMS PrecipFlag values
# ---------------------------------------------------------------------

PRECIPFLAG_NO_PRECIP = 0
PRECIPFLAG_WARM_STRATIFORM = 1
PRECIPFLAG_SNOW = 3
PRECIPFLAG_CONVECTION = 6
PRECIPFLAG_HAIL = 7
PRECIPFLAG_COOL_STRATIFORM = 10
PRECIPFLAG_TROPICAL_STRATIFORM = 91
PRECIPFLAG_TROPICAL_CONVECTION = 96


@dataclass
class ClassificationResult:
    phase: np.ndarray
    confidence: np.ndarray
    intensity: np.ndarray


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _finite(a: np.ndarray) -> np.ndarray:
    return np.isfinite(a)


def _valid_bright_band(
    bb_top_m: np.ndarray,
    bb_bottom_m: np.ndarray,
) -> np.ndarray:
    """
    Identify valid MRMS bright-band top/bottom values.

    Invalid / missing MRMS values are rejected.
    """
    return (
        np.isfinite(bb_top_m)
        & np.isfinite(bb_bottom_m)
        & (bb_top_m >= 0)
        & (bb_bottom_m >= 0)
        & (bb_top_m >= bb_bottom_m)
    )


def rain_intensity_dbz(dbz: np.ndarray) -> np.ndarray:
    """
    Convert reflectivity into a simple 0-5 intensity category.

    This is used only for the underlying rain classification.
    """
    out = np.zeros(dbz.shape, dtype=np.uint8)

    valid = np.isfinite(dbz)

    out[valid & (dbz >= 15) & (dbz < 25)] = 1
    out[valid & (dbz >= 25) & (dbz < 35)] = 2
    out[valid & (dbz >= 35) & (dbz < 45)] = 3
    out[valid & (dbz >= 45) & (dbz < 55)] = 4
    out[valid & (dbz >= 55)] = 5

    return out


def _precipitation_gate(
    reflectivity: np.ndarray,
    precip_flag: np.ndarray,
) -> np.ndarray:
    """
    Determine whether MRMS indicates precipitation.

    PrecipFlag is the primary discriminator.

    Reflectivity is retained as a fallback for cases where the flag
    is missing or invalid.
    """

    valid_flag = np.isfinite(precip_flag)

    # Any known non-zero MRMS precipitation flag means precipitation.
    flagged_precip = valid_flag & (precip_flag != PRECIPFLAG_NO_PRECIP)

    # If PrecipFlag is unavailable, use a conservative reflectivity
    # fallback.
    fallback = (~valid_flag) & np.isfinite(reflectivity) & (reflectivity >= 10.0)

    return flagged_precip | fallback


# ---------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------

def classify_initial(
    reflectivity: np.ndarray,
    precip_flag: np.ndarray,
    bb_top_m: np.ndarray,
    bb_bottom_m: np.ndarray,
    wetbulb_c: np.ndarray,
    freezing_level_m: np.ndarray,
    rqi: np.ndarray,
) -> ClassificationResult:

    phase = np.full(
        reflectivity.shape,
        CLEAR,
        dtype=np.uint8,
    )

    confidence = np.zeros(
        reflectivity.shape,
        dtype=np.float32,
    )

    # ---------------------------------------------------------------
    # 1. Determine precipitation
    # ---------------------------------------------------------------

    precip = _precipitation_gate(
        reflectivity,
        precip_flag,
    )

    # ---------------------------------------------------------------
    # 2. Environmental fields
    # ---------------------------------------------------------------

    valid_wb = np.isfinite(wetbulb_c)

    cold_surface = (
        valid_wb
        & (wetbulb_c <= 0.0)
    )

    very_cold_surface = (
        valid_wb
        & (wetbulb_c <= -1.0)
    )

    warm_surface = (
        valid_wb
        & (wetbulb_c > 0.5)
    )

    near_freezing_surface = (
        valid_wb
        & (wetbulb_c > -1.0)
        & (wetbulb_c <= 0.5)
    )

    # ---------------------------------------------------------------
    # 3. Bright-band information
    # ---------------------------------------------------------------

    has_bb = _valid_bright_band(
        bb_top_m,
        bb_bottom_m,
    )

    # A valid freezing-level field is useful as supporting evidence,
    # but a single freezing-level height is NOT sufficient to
    # confidently separate sleet from freezing rain.
    valid_freezing_level = np.isfinite(freezing_level_m)

    low_freezing_level = (
        valid_freezing_level
        & (freezing_level_m <= 1000.0)
    )

    # ---------------------------------------------------------------
    # 4. PrecipFlag information
    # ---------------------------------------------------------------

    valid_flag = np.isfinite(precip_flag)

    flag = np.where(
        valid_flag,
        precip_flag,
        -999,
    ).astype(np.int16)

    snow_flag = (
        valid_flag
        & (flag == PRECIPFLAG_SNOW)
    )

    warm_rain_flag = (
        valid_flag
        & (flag == PRECIPFLAG_WARM_STRATIFORM)
    )

    cool_stratiform_flag = (
        valid_flag
        & (flag == PRECIPFLAG_COOL_STRATIFORM)
    )

    winter_candidate_flag = (
        snow_flag
        | cool_stratiform_flag
    )

    # ---------------------------------------------------------------
    # 5. Rain
    #
    # Rain is deliberately transparent in render.py.
    # ---------------------------------------------------------------

    rain = precip & (
        warm_surface
        | warm_rain_flag
    )

    phase[rain] = RAIN

    confidence[rain] = 0.80

    # ---------------------------------------------------------------
    # 6. Strong snow signal
    #
    # Explicit MRMS snow flag + cold surface is our strongest
    # first-pass winter classification.
    # ---------------------------------------------------------------

    snow = (
        precip
        & cold_surface
        & snow_flag
    )

    phase[snow] = SNOW
    confidence[snow] = 0.82

    # ---------------------------------------------------------------
    # 7. Very cold precipitation
    #
    # Very cold surface conditions strongly support snow when MRMS
    # has not explicitly identified rain.
    # ---------------------------------------------------------------

    very_cold_snow = (
        precip
        & very_cold_surface
        & ~rain
        & ~snow_flag
        & ~has_bb
    )

    phase[very_cold_snow] = SNOW
    confidence[very_cold_snow] = 0.72

    # ---------------------------------------------------------------
    # 8. Cold precipitation with bright band
    #
    # This is deliberately classified as MIXED rather than trying
    # to force a sleet/FZRA decision.
    # ---------------------------------------------------------------

    cold_mixed = (
        precip
        & cold_surface
        & has_bb
        & ~rain
    )

    phase[cold_mixed] = MIXED
    confidence[cold_mixed] = 0.58

    # ---------------------------------------------------------------
    # 9. Cool stratiform precipitation
    #
    # MRMS cool-stratiform precipitation near/below freezing is
    # treated as winter precipitation, but conservatively left as
    # MIXED unless the snow signal is stronger.
    # ---------------------------------------------------------------

    cool_winter = (
        precip
        & near_freezing_surface
        & cool_stratiform_flag
        & ~rain
        & ~snow
    )

    phase[cool_winter] = MIXED
    confidence[cool_winter] = 0.55

    # ---------------------------------------------------------------
    # 10. Near-freezing precipitation without a clean phase signal
    #
    # Do NOT make these transparent. They are exactly the cases
    # where the eventual vertical thermal/dual-pol logic needs to
    # operate.
    # ---------------------------------------------------------------

    uncertain_winter = (
        precip
        & near_freezing_surface
        & ~rain
        & ~snow
        & ~cold_mixed
        & ~cool_winter
    )

    phase[uncertain_winter] = UNKNOWN
    confidence[uncertain_winter] = 0.35

    # ---------------------------------------------------------------
    # 11. Missing thermodynamic information
    #
    # If precipitation exists but wet-bulb temperature is unavailable,
    # don't silently call it rain.
    #
    # Keep it as UNKNOWN so winter precipitation isn't accidentally
    # made transparent.
    # ---------------------------------------------------------------

    missing_environment = (
        precip
        & ~valid_wb
        & ~rain
        & ~snow
    )

    phase[missing_environment] = UNKNOWN
    confidence[missing_environment] = 0.25

    # ---------------------------------------------------------------
    # 12. RQI adjustment
    # ---------------------------------------------------------------

    good_rqi = np.clip(
        np.nan_to_num(
            rqi,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ),
        0.0,
        1.0,
    )

    # RQI affects confidence, not phase.
    confidence *= 0.50 + 0.50 * good_rqi

    # ---------------------------------------------------------------
    # 13. Rain intensity
    # ---------------------------------------------------------------

    intensity = rain_intensity_dbz(
        reflectivity
    )

    # Only actual rain gets rain intensity.
    intensity[phase != RAIN] = 0

    # ---------------------------------------------------------------
    # 14. Explicit clear pixels
    # ---------------------------------------------------------------

    no_precip = ~precip

    phase[no_precip] = CLEAR
    confidence[no_precip] = 0.0
    intensity[no_precip] = 0

    return ClassificationResult(
        phase=phase,
        confidence=confidence,
        intensity=intensity,
    )
