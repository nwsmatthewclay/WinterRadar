from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ClassificationResult:
    phase: np.ndarray
    confidence: np.ndarray
    intensity: np.ndarray


# Phase codes used in the raster and JSON output.
CLEAR = 0
RAIN = 1
SNOW = 2
SLEET = 3
FZRA = 4
MIXED = 5
UNKNOWN = 9


def rain_intensity_dbz(dbz: np.ndarray) -> np.ndarray:
    """Rain intensity class: 0 none, then 1-5 increasing intensity."""
    out = np.zeros(dbz.shape, dtype=np.uint8)
    out[(dbz >= 15) & (dbz < 25)] = 1
    out[(dbz >= 25) & (dbz < 35)] = 2
    out[(dbz >= 35) & (dbz < 45)] = 3
    out[(dbz >= 45) & (dbz < 55)] = 4
    out[dbz >= 55] = 5
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
    """Initial radar-assisted winter classifier.

    This is deliberately conservative. It does not claim to replace a full
    vertical thermodynamic precipitation-type algorithm yet.
    """
    phase = np.full(reflectivity.shape, CLEAR, dtype=np.uint8)
    confidence = np.zeros(reflectivity.shape, dtype=np.float32)

    precip = reflectivity >= 10
    phase[precip] = RAIN

    valid_wb = np.isfinite(wetbulb_c) & (wetbulb_c > -90)
    cold_surface = valid_wb & (wetbulb_c <= 0.0)
    warm_surface = valid_wb & (wetbulb_c > 0.5)

    # MRMS PrecipFlag contains broad surface-precipitation context. We treat
    # its snow flag as a strong prior, but not as the final surface-phase answer.
    snow_flag = np.isin(precip_flag.astype(np.int16), [4, 5])

    # Bright band presence: positive top/bottom heights are the first proxy for
    # a melting layer in this prototype.
    has_bb = (bb_top_m > 0) & (bb_bottom_m > 0) & (bb_top_m >= bb_bottom_m)
    bb_depth = np.where(has_bb, bb_top_m - bb_bottom_m, 0.0)

    # Snow: cold surface and snow/limited melting evidence.
    snow = precip & cold_surface & (snow_flag | ~has_bb)
    phase[snow] = SNOW
    confidence[snow] = 0.72

    # Shallow-below-zero surface with a detected melting layer: mixed winter
    # precipitation until a vertical thermal profile is available.
    mixed = precip & cold_surface & has_bb
    phase[mixed] = MIXED
    confidence[mixed] = 0.55

    # When the surface is warm enough, retain rain even if a melting layer exists.
    rain = precip & warm_surface
    phase[rain] = RAIN
    confidence[rain] = 0.70

    # Ice/sleet are deliberately represented as MIXED in this first pass.
    # The HRRR/RAP profile module will split this using warm/cold layer energy.
    
    # RQI reduces confidence where radar coverage is poor.
    good_rqi = np.clip(np.nan_to_num(rqi, nan=0.0), 0.0, 1.0)
    confidence *= (0.5 + 0.5 * good_rqi)

    intensity = rain_intensity_dbz(reflectivity)
    intensity[(phase != RAIN) & (reflectivity < 10)] = 0

    # Clean obvious non-precipitation/no-coverage pixels.
    bad = ~precip | (reflectivity <= -10)
    phase[bad] = CLEAR
    confidence[bad] = 0.0

    return ClassificationResult(phase, confidence, intensity)
