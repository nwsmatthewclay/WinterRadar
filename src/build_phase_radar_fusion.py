#!/usr/bin/env python3
"""
WinterRadar radar/dual-pol phase evidence fusion.

Purpose
-------
Create a conservative observational evidence layer that complements the
existing RAP vertical-profile precipitation-phase engine. This script DOES
NOT replace or alter the primary phase classification.

Inputs
------
- MRMS lowest-altitude reflectivity
- 3-D MergedRhoHV at 0.50-4.00 km
- 3-D MergedZdr at 0.50-4.00 km
- Existing outputs/phase_probabilities.npz produced by main.py
- Existing outputs/mrms_current.json for the RAP-domain placement

Outputs
-------
- outputs/phase_radar_fusion.png
- outputs/phase_radar_fusion.json

The radar evidence is intentionally conservative:
- Melting-layer candidate: rhoHV <= 0.95 AND ZDR >= +0.5 dB
- Dry-snow-like candidate: rhoHV >= 0.99 AND ZDR <= 0.0 dB

These are observational signatures, not stand-alone precipitation-type
classifications. The existing RAP/Bourgouin result remains the primary phase
solution.
"""

from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

# Keep the GitHub Actions memory footprint predictable.
for key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(key, "1")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pygrib
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
DUALPOL_DIR = DATA_DIR / "dualpol"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LEVELS_KM = (0.50, 1.00, 1.50, 2.00, 2.50, 3.00, 3.50, 4.00)
DOWNSAMPLE = 5
TARGET_SHAPE = (700, 1400)
PRECIP_DBZ = 10.0

# Conservative evidence thresholds. These are deliberately stricter than the
# exploratory winter-mask diagnostic that uses a broad rhoHV/ZDR window.
ML_RHO_MAX = 0.95
ML_ZDR_MIN = 0.5
ML_ZDR_MAX = 5.0
SNOW_RHO_MIN = 0.99
SNOW_ZDR_MIN = -2.0
SNOW_ZDR_MAX = 0.0

ML_FRACTION_THRESHOLD = 0.25
SNOW_FRACTION_THRESHOLD = 0.50
VALID_FRACTION_THRESHOLD = 0.50


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def level_text(level: float) -> str:
    return f"{float(level):05.2f}"


def dualpol_path(product: str, level: float) -> Path:
    text = level_text(level)
    return DUALPOL_DIR / f"{text}km" / f"MRMS_{product}_{text}.latest.grib2"


def read_2d_grib(path: Path) -> np.ndarray:
    """Read one MRMS GRIB message with pygrib and release native handles immediately.

    The fusion step used to open each field through xarray/cfgrib. That path
    was scientifically fine, but repeated cfgrib/eccodes teardown after the
    16 dual-pol files could trigger a native ``double free or corruption``
    abort on GitHub Actions. The diagnostic builders already use pygrib
    safely, so fusion now follows the same one-file/one-handle pattern.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    grbs = None
    grb = None
    try:
        grbs = pygrib.open(str(path))
        grb = grbs.message(1)
        data = np.array(grb.values, dtype=np.float32, copy=True)
    finally:
        grb = None
        if grbs is not None:
            try:
                grbs.close()
            except Exception:
                pass
        grbs = None
        gc.collect()

    if data.ndim != 2:
        raise RuntimeError(f"Expected 2-D field in {path}; got {data.shape}")

    data[~np.isfinite(data)] = np.nan
    data[data <= -90.0] = np.nan
    return data


def block_mean_bool(mask: np.ndarray, factor: int = DOWNSAMPLE) -> np.ndarray:
    """Block-average a boolean field without retaining extra full-res arrays."""
    rows, cols = mask.shape
    h = rows // factor
    w = cols // factor
    trimmed = mask[: h * factor, : w * factor]
    return trimmed.reshape(h, factor, w, factor).mean(axis=(1, 3)).astype(np.float32)


def block_max_bool(mask: np.ndarray, factor: int = DOWNSAMPLE) -> np.ndarray:
    rows, cols = mask.shape
    h = rows // factor
    w = cols // factor
    trimmed = mask[: h * factor, : w * factor]
    return trimmed.reshape(h, factor, w, factor).max(axis=(1, 3))


def block_mean_float(data: np.ndarray, factor: int = DOWNSAMPLE) -> np.ndarray:
    rows, cols = data.shape
    h = rows // factor
    w = cols // factor
    trimmed = data[: h * factor, : w * factor]
    return np.nanmean(trimmed.reshape(h, factor, w, factor), axis=(1, 3)).astype(np.float32)


# ---------------------------------------------------------------------------
# Existing RAP phase solution placement
# ---------------------------------------------------------------------------

def load_phase_probabilities() -> tuple[dict[str, np.ndarray], dict]:
    path = OUTPUT_DIR / "phase_probabilities.npz"
    meta_path = OUTPUT_DIR / "mrms_current.json"

    if not path.exists():
        raise FileNotFoundError(path)
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    with np.load(path) as npz:
        required = ("rain", "snow", "sleet", "freezing_rain")
        data = {name: np.asarray(npz[name], dtype=np.float32).copy() for name in required}

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return data, meta


def place_phase_probabilities(
    phase_data: dict[str, np.ndarray],
    metadata: dict,
    full_shape: tuple[int, int],
    factor: int = DOWNSAMPLE,
) -> dict[str, np.ndarray]:
    """
    Expand the reduced RAP-domain probability arrays into a full-CONUS
    reduced grid. The arrays generated by the successful RAP engine live on
    the domain described in phase_diagnostics.phase_domain_mrms_indices.
    """
    full_rows, full_cols = full_shape
    out_rows = full_rows // factor
    out_cols = full_cols // factor
    placed = {
        name: np.full((out_rows, out_cols), np.nan, dtype=np.float32)
        for name in phase_data
    }

    diag = metadata.get("phase_diagnostics") or {}
    idx = diag.get("phase_domain_mrms_indices") or {}

    if all(k in idx for k in ("row_start", "row_end", "col_start", "col_end")):
        # Domain-only diagnostic arrays: place them using their native MRMS
        # indices.
        y0 = max(0, int(idx["row_start"]) // factor)
        x0 = max(0, int(idx["col_start"]) // factor)
        domain_rows = max(0, int(idx["row_end"]) - int(idx["row_start"]))
        domain_cols = max(0, int(idx["col_end"]) - int(idx["col_start"]))
        expected_h = max(1, int(np.ceil(domain_rows / factor)))
        expected_w = max(1, int(np.ceil(domain_cols / factor)))

        def nearest_resize(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
            if arr.shape == shape:
                return arr
            rr = np.minimum((np.arange(shape[0]) * arr.shape[0] // shape[0]), arr.shape[0] - 1)
            cc = np.minimum((np.arange(shape[1]) * arr.shape[1] // shape[1]), arr.shape[1] - 1)
            return arr[np.ix_(rr, cc)]

        for name, arr in phase_data.items():
            arr2 = nearest_resize(np.asarray(arr, dtype=np.float32), (expected_h, expected_w))
            h = min(arr2.shape[0], out_rows - y0)
            w = min(arr2.shape[1], out_cols - x0)
            if h > 0 and w > 0:
                placed[name][y0:y0 + h, x0:x0 + w] = arr2[:h, :w]
    else:
        # Current main.py writes a FULL-CONUS 10x diagnostic grid (350x700),
        # while the dual-pol evidence is a 5x grid (700x1400). The old code
        # placed the 350x700 array into the upper-left corner of the 700x1400
        # grid, which put the RAP phase field over the wrong geography and
        # caused phase agreement to have essentially no overlapping valid
        # model/radar pixels. Expand the full-CONUS probability grid to the
        # evidence grid with nearest-neighbor replication.
        y0 = 0
        x0 = 0

        def nearest_resize(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
            if arr.shape == shape:
                return arr
            rr = np.minimum((np.arange(shape[0]) * arr.shape[0] // shape[0]), arr.shape[0] - 1)
            cc = np.minimum((np.arange(shape[1]) * arr.shape[1] // shape[1]), arr.shape[1] - 1)
            return arr[np.ix_(rr, cc)]

        for name, arr in phase_data.items():
            arr2 = nearest_resize(np.asarray(arr, dtype=np.float32), (out_rows, out_cols))
            placed[name][:] = arr2

    return placed


def dominant_phase(prob: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    names = ("rain", "snow", "sleet", "freezing_rain")
    stack = np.stack([prob[n] for n in names], axis=0)
    safe = np.nan_to_num(stack, nan=-1.0)
    order = np.argsort(safe, axis=0)
    top_idx = order[-1]
    second_idx = order[-2]
    top = np.take_along_axis(safe, top_idx[None, ...], axis=0)[0]
    second = np.take_along_axis(safe, second_idx[None, ...], axis=0)[0]
    valid = np.isfinite(stack).any(axis=0)

    # Mirror the phase engine's conservative threshold for a clearly dominant
    # type. Unclear columns are encoded as -1.
    dominant = np.where(
        valid & (top >= 55.0) & ((top - second) >= 10.0),
        top_idx,
        -1,
    ).astype(np.int8)
    return dominant, top, second


# ---------------------------------------------------------------------------
# Radar evidence
# ---------------------------------------------------------------------------

AGREEMENT_NONE = 0
AGREEMENT_CONSISTENT = 1
AGREEMENT_CONFLICT = 2
AGREEMENT_RADAR_CONFLICT = 3
AGREEMENT_NO_DECISIVE_RADAR = 4


def build_phase_agreement(
    evidence: dict[str, np.ndarray],
    dominant: np.ndarray,
    top: np.ndarray,
    second: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict]:
    """Compare the primary RAP phase solution with independent radar evidence.

    The comparison deliberately avoids claiming that dual-pol alone can
    distinguish sleet from freezing rain. It asks whether the radar evidence
    is consistent with, conflicts with, or cannot meaningfully evaluate the
    thermodynamic phase solution.
    """
    precip = np.asarray(evidence["precip"], dtype=bool)
    valid_radar = precip & (evidence["valid_fraction"] >= VALID_FRACTION_THRESHOLD)
    ml = evidence["ml_fraction"] >= ML_FRACTION_THRESHOLD
    dry = evidence["dry_snow_fraction"] >= SNOW_FRACTION_THRESHOLD
    radar_both = ml & dry

    model_valid = dominant >= 0
    model_rain = dominant == 0
    model_snow = dominant == 1
    model_ice = np.isin(dominant, [2, 3])

    agreement = np.zeros(dominant.shape, dtype=np.uint8)

    # Conflicting vertical radar signatures get their own category.
    agreement[valid_radar & radar_both] = AGREEMENT_RADAR_CONFLICT

    available = valid_radar & ~radar_both & model_valid

    consistent = (
        (available & model_snow & dry)
        | (available & model_ice & ml)
        | (available & model_rain & ~ml & ~dry)
    )

    conflict = (
        (available & model_snow & ml)
        | (available & model_ice & dry)
        | (available & model_rain & (ml | dry))
    )

    agreement[consistent] = AGREEMENT_CONSISTENT
    agreement[conflict] = AGREEMENT_CONFLICT

    # A valid dual-pol column with no decisive signature is not a disagreement.
    agreement[valid_radar & ~radar_both & model_valid & ~consistent & ~conflict] = AGREEMENT_NO_DECISIVE_RADAR

    # Model ambiguity / missing radar information remains unclassified.
    score = np.zeros_like(dominant, dtype=np.float32)
    score[consistent] = np.minimum(1.0, np.maximum(
        evidence["dry_snow_fraction"][consistent],
        evidence["ml_fraction"][consistent],
    ))
    score[conflict] = np.minimum(1.0, np.maximum(
        evidence["dry_snow_fraction"][conflict],
        evidence["ml_fraction"][conflict],
    ))
    score[agreement == AGREEMENT_NO_DECISIVE_RADAR] = evidence["valid_fraction"][agreement == AGREEMENT_NO_DECISIVE_RADAR]
    score[agreement == AGREEMENT_RADAR_CONFLICT] = np.minimum(
        1.0,
        np.maximum(
            evidence["ml_fraction"][agreement == AGREEMENT_RADAR_CONFLICT],
            evidence["dry_snow_fraction"][agreement == AGREEMENT_RADAR_CONFLICT],
        ),
    )

    stats = {
        "usable_precip_area_percent": float(100.0 * np.count_nonzero(valid_radar) / max(np.count_nonzero(precip), 1)),
        "consistent_area_percent": float(100.0 * np.count_nonzero(agreement == AGREEMENT_CONSISTENT) / max(np.count_nonzero(valid_radar), 1)),
        "conflict_area_percent": float(100.0 * np.count_nonzero(agreement == AGREEMENT_CONFLICT) / max(np.count_nonzero(valid_radar), 1)),
        "radar_conflict_area_percent": float(100.0 * np.count_nonzero(agreement == AGREEMENT_RADAR_CONFLICT) / max(np.count_nonzero(valid_radar), 1)),
        "no_decisive_radar_area_percent": float(100.0 * np.count_nonzero(agreement == AGREEMENT_NO_DECISIVE_RADAR) / max(np.count_nonzero(valid_radar), 1)),
        "precip_pixels": int(np.count_nonzero(precip)),
        "valid_radar_pixels": int(np.count_nonzero(valid_radar)),
        "model_valid_pixels": int(np.count_nonzero(model_valid)),
        "overlap_pixels": int(np.count_nonzero(valid_radar & model_valid)),
        "consistent_pixels": int(np.count_nonzero(agreement == AGREEMENT_CONSISTENT)),
        "conflict_pixels": int(np.count_nonzero(agreement == AGREEMENT_CONFLICT)),
        "radar_conflict_pixels": int(np.count_nonzero(agreement == AGREEMENT_RADAR_CONFLICT)),
        "no_decisive_radar_pixels": int(np.count_nonzero(agreement == AGREEMENT_NO_DECISIVE_RADAR)),
        "unclassified_pixels": int(np.count_nonzero(agreement == AGREEMENT_NONE)),
    }

    return {
        "agreement": agreement,
        "strength": score,
    }, stats


def build_radar_evidence(reflectivity: np.ndarray) -> tuple[dict[str, np.ndarray], list[dict]]:
    """
    Produce reduced-resolution vertical evidence fields.

    We only retain compact 700x1400 arrays, avoiding a 16-field full-CONUS
    volume in memory.
    """
    precip_full = np.isfinite(reflectivity) & (reflectivity >= PRECIP_DBZ)
    precip = block_mean_bool(precip_full) >= 0.10
    del precip_full

    out_shape = precip.shape
    ml_sum = np.zeros(out_shape, dtype=np.float32)
    snow_sum = np.zeros(out_shape, dtype=np.float32)
    valid_sum = np.zeros(out_shape, dtype=np.float32)
    level_summaries: list[dict] = []

    for level in LEVELS_KM:
        rho_path = dualpol_path("MergedRhoHV", level)
        zdr_path = dualpol_path("MergedZdr", level)

        print(f"Loading vertical dual-pol evidence @ {level:.2f} km", flush=True)

        rho = read_2d_grib(rho_path)
        zdr = read_2d_grib(zdr_path)

        if rho.shape != reflectivity.shape or zdr.shape != reflectivity.shape:
            raise RuntimeError(
                f"Dual-pol shape mismatch at {level:.2f} km: "
                f"rho={rho.shape}, zdr={zdr.shape}, ref={reflectivity.shape}"
            )

        valid = np.isfinite(rho) & np.isfinite(zdr)
        ml = (
            valid
            & (rho <= ML_RHO_MAX)
            & (zdr >= ML_ZDR_MIN)
            & (zdr <= ML_ZDR_MAX)
        )
        dry_snow = (
            valid
            & (rho >= SNOW_RHO_MIN)
            & (zdr >= SNOW_ZDR_MIN)
            & (zdr <= SNOW_ZDR_MAX)
        )

        valid_d = block_mean_bool(valid)
        ml_d = block_mean_bool(ml)
        snow_d = block_mean_bool(dry_snow)

        ml_sum += ml_d
        snow_sum += snow_d
        valid_sum += valid_d

        level_summaries.append({
            "level_km": float(level),
            "valid_fraction_mean_percent": float(np.mean(valid_d) * 100.0),
            "melting_layer_candidate_area_percent": float(np.mean((ml_d >= 0.10) & precip) * 100.0),
            "dry_snow_candidate_area_percent": float(np.mean((snow_d >= 0.10) & precip) * 100.0),
        })

        del rho, zdr, valid, ml, dry_snow, valid_d, ml_d, snow_d
        gc.collect()

    ml_fraction = ml_sum / float(len(LEVELS_KM))
    snow_fraction = snow_sum / float(len(LEVELS_KM))
    valid_fraction = valid_sum / float(len(LEVELS_KM))

    usable = precip & (valid_fraction >= VALID_FRACTION_THRESHOLD)
    melting = usable & (ml_fraction >= ML_FRACTION_THRESHOLD)
    dry_snow = usable & (snow_fraction >= SNOW_FRACTION_THRESHOLD)
    both = melting & dry_snow

    # Give precedence to the dual-signature category so it can be inspected as
    # a potentially transitional/complex column instead of hiding one signal.
    category = np.zeros(out_shape, dtype=np.uint8)
    category[melting & ~dry_snow] = 1
    category[dry_snow & ~melting] = 2
    category[both] = 3
    category[(precip) & (valid_fraction < VALID_FRACTION_THRESHOLD)] = 4

    return {
        "precip": precip,
        "ml_fraction": ml_fraction,
        "dry_snow_fraction": snow_fraction,
        "valid_fraction": valid_fraction,
        "category": category,
    }, level_summaries


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def load_lat_lon_bounds(metadata: dict, shape: tuple[int, int]) -> tuple[float, float, float, float]:
    bounds = metadata.get("bounds")
    if isinstance(bounds, list) and len(bounds) == 4:
        south, west, north, east = map(float, bounds)
        return south, west, north, east
    # Full-CONUS fallback used by the current MRMS native grid.
    return 20.005001, -129.995, 54.995, -60.005002


def resize_nearest_rgba(arr: np.ndarray, out_size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(arr, mode="RGBA")
    image = image.resize(out_size, Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.uint8)


def make_map_overlay(evidence: dict[str, np.ndarray]) -> np.ndarray:
    """Create a sparse RGBA evidence overlay on the reduced grid."""
    h, w = evidence["category"].shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    cat = evidence["category"]

    # 1 = melting-layer candidate; 2 = dry-snow-like candidate;
    # 3 = both; 4 = data limited.
    rgba[cat == 1] = (230, 70, 180, 170)
    rgba[cat == 2] = (70, 130, 235, 145)
    rgba[cat == 3] = (160, 80, 220, 185)
    rgba[cat == 4] = (130, 130, 130, 85)
    return rgba


def make_agreement_overlay(evaluation: dict[str, np.ndarray]) -> np.ndarray:
    category = evaluation["agreement"]
    strength = np.clip(evaluation["strength"], 0.0, 1.0)
    rgba = np.zeros((*category.shape, 4), dtype=np.uint8)

    # Green = model/radar consistency; red = disagreement; orange = conflicting
    # radar signatures; gray = valid but no decisive radar signature.
    alpha_consistent = np.rint(90 + 110 * strength).astype(np.uint8)
    alpha_conflict = np.rint(95 + 105 * strength).astype(np.uint8)
    alpha_both = np.rint(105 + 95 * strength).astype(np.uint8)

    rgba[category == AGREEMENT_CONSISTENT, :3] = (50, 170, 95)
    rgba[category == AGREEMENT_CONSISTENT, 3] = alpha_consistent[category == AGREEMENT_CONSISTENT]

    rgba[category == AGREEMENT_CONFLICT, :3] = (220, 65, 65)
    rgba[category == AGREEMENT_CONFLICT, 3] = alpha_conflict[category == AGREEMENT_CONFLICT]

    rgba[category == AGREEMENT_RADAR_CONFLICT, :3] = (225, 145, 35)
    rgba[category == AGREEMENT_RADAR_CONFLICT, 3] = alpha_both[category == AGREEMENT_RADAR_CONFLICT]

    rgba[category == AGREEMENT_NO_DECISIVE_RADAR] = (125, 135, 145, 70)
    return rgba


def make_agreement_qc_image(
    evaluation: dict[str, np.ndarray],
    agreement_stats: dict,
    metadata: dict,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), constrained_layout=True)

    category = evaluation["agreement"]
    strength = evaluation["strength"]

    im0 = axes[0].imshow(
        category,
        origin="upper",
        vmin=0,
        vmax=4,
        interpolation="nearest",
        aspect="auto",
    )
    axes[0].set_title("Phase Agreement Category")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    fig.colorbar(im0, ax=axes[0], shrink=0.82, ticks=[0, 1, 2, 3, 4], label="category")

    im1 = axes[1].imshow(
        strength,
        origin="upper",
        vmin=0,
        vmax=1,
        interpolation="nearest",
        aspect="auto",
    )
    axes[1].set_title("Agreement / Evidence Strength")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    fig.colorbar(im1, ax=axes[1], shrink=0.82, label="0–1")

    rap_valid = (metadata.get("phase_diagnostics") or {}).get("rap_valid_time_utc", "unknown")
    fig.suptitle(
        "WinterRadar Phase Agreement Diagnostic\n"
        f"RAP/Bourgouin: {rap_valid} • "
        f"Consistent {agreement_stats.get('consistent_area_percent', 0.0):.1f}% • "
        f"Conflicting {agreement_stats.get('conflict_area_percent', 0.0):.1f}% of usable precipitation",
        fontsize=15,
        fontweight="bold",
    )

    fig.savefig(
        OUTPUT_DIR / "phase_agreement.png",
        dpi=130,
        facecolor="white",
        bbox_inches="tight",
    )
    plt.close(fig)


def make_qc_image(
    evidence: dict[str, np.ndarray],
    dominant: np.ndarray,
    top: np.ndarray,
    second: np.ndarray,
    level_summaries: list[dict],
    metadata: dict,
    evaluation: dict[str, np.ndarray],
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True)

    panels = [
        ("Melting-Layer Evidence", evidence["ml_fraction"], 0, 1, "fraction of 8 levels"),
        ("Dry-Snow-Like Evidence", evidence["dry_snow_fraction"], 0, 1, "fraction of 8 levels"),
        ("Valid Vertical Dual-Pol", evidence["valid_fraction"], 0, 1, "fraction of 8 levels"),
        ("Radar Evidence Category", evidence["category"], 0, 4, "category"),
        ("Phase Agreement", evaluation["agreement"], 0, 4, "category"),
        ("Agreement Strength", evaluation["strength"], 0, 1, "0–1"),
    ]

    for ax, (title, data, vmin, vmax, label) in zip(axes.flat, panels):
        im = ax.imshow(data, origin="upper", vmin=vmin, vmax=vmax, interpolation="nearest", aspect="auto")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.82, label=label)

    rap_valid = (metadata.get("phase_diagnostics") or {}).get("rap_valid_time_utc", "unknown")
    fig.suptitle(
        "WinterRadar Radar Phase Evidence Fusion\n"
        f"RAP solution: {rap_valid} • 3-D MRMS RHOHV/ZDR: {len(level_summaries)} levels",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(OUTPUT_DIR / "phase_radar_fusion.png", dpi=130, facecolor="white", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# JSON summary
# ---------------------------------------------------------------------------

def pct(mask: np.ndarray, denom: np.ndarray | None = None) -> float:
    if denom is None:
        finite = np.isfinite(mask)
        return float(100.0 * np.count_nonzero(mask[finite]) / max(np.count_nonzero(finite), 1))
    return float(100.0 * np.count_nonzero(mask & denom) / max(np.count_nonzero(denom), 1))


def write_summary(
    evidence: dict[str, np.ndarray],
    dominant: np.ndarray,
    top: np.ndarray,
    second: np.ndarray,
    level_summaries: list[dict],
    metadata: dict,
    agreement_stats: dict,
) -> None:
    precip = evidence["precip"]
    usable = precip & (evidence["valid_fraction"] >= VALID_FRACTION_THRESHOLD)
    category = evidence["category"]

    phase_names = {
        0: "rain",
        1: "snow",
        2: "sleet",
        3: "freezing_rain",
    }

    phase_summary = {}
    for code, name in phase_names.items():
        phase_area = usable & (dominant == code)
        support = phase_area & (
            ((code == 1) & (evidence["dry_snow_fraction"] >= SNOW_FRACTION_THRESHOLD))
            | ((code in (2, 3)) & (evidence["ml_fraction"] >= ML_FRACTION_THRESHOLD))
            | ((code == 0) & (evidence["ml_fraction"] < ML_FRACTION_THRESHOLD))
        )
        phase_summary[name] = {
            "dominant_area_percent_of_usable_precip": float(
                100.0 * np.count_nonzero(phase_area) / max(np.count_nonzero(usable), 1)
            ),
            "radar_evidence_consistent_area_percent": float(
                100.0 * np.count_nonzero(support) / max(np.count_nonzero(phase_area), 1)
            ),
            "mean_probability_percent": float(np.nanmean(np.where(phase_area, top, np.nan))) if np.any(phase_area) else None,
        }

    qc = {
        "status": "ok",
        "purpose": "Observational radar/dual-pol evidence layer; primary phase classification is unchanged.",
        "primary_phase_engine": (metadata.get("phase_diagnostics") or {}).get("engine", metadata.get("phase_status", "unknown")),
        "mrms_time_utc": metadata.get("mrms_time_utc"),
        "rap_valid_time_utc": (metadata.get("phase_diagnostics") or {}).get("rap_valid_time_utc"),
        "radar_evidence_thresholds": {
            "precip_dbz": PRECIP_DBZ,
            "melting_layer": {
                "rhohv_max": ML_RHO_MAX,
                "zdr_min_db": ML_ZDR_MIN,
                "zdr_max_db": ML_ZDR_MAX,
                "vertical_fraction_threshold": ML_FRACTION_THRESHOLD,
            },
            "dry_snow_like": {
                "rhohv_min": SNOW_RHO_MIN,
                "zdr_min_db": SNOW_ZDR_MIN,
                "zdr_max_db": SNOW_ZDR_MAX,
                "vertical_fraction_threshold": SNOW_FRACTION_THRESHOLD,
            },
            "minimum_valid_vertical_fraction": VALID_FRACTION_THRESHOLD,
        },
        "grid": {
            "source_shape": [3500, 7000],
            "diagnostic_shape": list(category.shape),
            "downsample_factor": DOWNSAMPLE,
        },
        "precipitation_area_percent": float(100.0 * np.mean(precip)),
        "usable_precipitation_area_percent": float(100.0 * np.mean(usable)),
        "evidence_area_percent_of_usable_precip": {
            "melting_layer": float(100.0 * np.count_nonzero((category == 1) & usable) / max(np.count_nonzero(usable), 1)),
            "dry_snow_like": float(100.0 * np.count_nonzero((category == 2) & usable) / max(np.count_nonzero(usable), 1)),
            "both": float(100.0 * np.count_nonzero((category == 3) & usable) / max(np.count_nonzero(usable), 1)),
            "data_limited": float(100.0 * np.count_nonzero((category == 4) & precip) / max(np.count_nonzero(precip), 1)),
        },
        "phase_context": phase_summary,
        "phase_agreement": agreement_stats,
        "mean_dominant_probability_percent": float(np.nanmean(np.where(usable, top, np.nan))) if np.any(usable) else None,
        "mean_runner_up_probability_percent": float(np.nanmean(np.where(usable, second, np.nan))) if np.any(usable) else None,
        "vertical_levels": level_summaries,
        "legend": {
            "1": "Melting-layer candidate",
            "2": "Dry-snow-like candidate",
            "3": "Both signatures present",
            "4": "Vertical dual-pol data limited",
        },
    }

    (OUTPUT_DIR / "phase_radar_fusion.json").write_text(
        json.dumps(qc, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    print("=" * 72)
    print("WINTER RADAR — RADAR PHASE EVIDENCE FUSION")
    print("=" * 72)

    # Load reflectivity through the project's canonical reader.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from read_mrms import get_values

    ref_path = DATA_DIR / "MRMS_ReflectivityAtLowestAltitude.latest.grib2"
    reflectivity, _, _ = get_values(ref_path, product="ReflectivityAtLowestAltitude")
    if reflectivity.shape != (3500, 7000):
        print(f"Warning: expected native MRMS (3500,7000), got {reflectivity.shape}")

    phase_data, metadata = load_phase_probabilities()
    placed_phase = place_phase_probabilities(phase_data, metadata, reflectivity.shape)
    dominant, top, second = dominant_phase(placed_phase)

    evidence, level_summaries = build_radar_evidence(reflectivity)
    evaluation, agreement_stats = build_phase_agreement(evidence, dominant, top, second)

    make_qc_image(evidence, dominant, top, second, level_summaries, metadata, evaluation)
    make_agreement_qc_image(evaluation, agreement_stats, metadata)
    write_summary(evidence, dominant, top, second, level_summaries, metadata, agreement_stats)

    reduced_rgba = make_map_overlay(evidence)
    full_rgba = resize_nearest_rgba(reduced_rgba, (reflectivity.shape[1], reflectivity.shape[0]))
    native_fusion_path = OUTPUT_DIR / "phase_radar_fusion_overlay.png"
    Image.fromarray(full_rgba, mode="RGBA").save(native_fusion_path, optimize=True)

    agreement_reduced = make_agreement_overlay(evaluation)
    agreement_full = resize_nearest_rgba(agreement_reduced, (reflectivity.shape[1], reflectivity.shape[0]))
    native_agreement_path = OUTPUT_DIR / "phase_agreement_overlay.png"
    Image.fromarray(agreement_full, mode="RGBA").save(native_agreement_path, optimize=True)

    # The main radar and winter-phase browser layers are Web Mercator copies.
    # Project these evidence overlays with the exact same routine so their
    # pixel geometry matches the live radar/phase imagery instead of placing a
    # native latitude/longitude raster over a Web Mercator raster.
    from project_mrms_webmercator import project_image

    bounds = load_lat_lon_bounds(metadata, reflectivity.shape)
    project_image(
        native_fusion_path,
        OUTPUT_DIR / "phase_radar_fusion_overlay_web.png",
        bounds,
    )
    project_image(
        native_agreement_path,
        OUTPUT_DIR / "phase_agreement_overlay_web.png",
        bounds,
    )

    agreement_payload = {
        "status": "ok",
        "purpose": "Compare the primary RAP/Bourgouin phase solution with independent MRMS vertical dual-pol evidence.",
        "primary_phase_engine": (metadata.get("phase_diagnostics") or {}).get("engine", metadata.get("phase_status", "unknown")),
        "mrms_time_utc": metadata.get("mrms_time_utc"),
        "rap_valid_time_utc": (metadata.get("phase_diagnostics") or {}).get("rap_valid_time_utc"),
        "categories": {
            "0": "No assessment / insufficient data",
            "1": "Phase solution consistent with available radar evidence",
            "2": "Radar evidence conflicts with phase solution",
            "3": "Conflicting vertical radar signatures",
            "4": "Radar usable, but no decisive phase signature",
        },
        "stats": agreement_stats,
        "overlay_file": "phase_agreement_overlay_web.png",
        "native_overlay_file": "phase_agreement_overlay.png",
        "web_projection": "EPSG:3857",
    }
    (OUTPUT_DIR / "phase_agreement.json").write_text(json.dumps(agreement_payload, indent=2), encoding="utf-8")
    # Compact browser-facing point-diagnostic grid.  This intentionally uses a
    # second nearest-neighbor reduction from the 350x700 agreement grid so the
    # browser can provide useful click diagnostics without shipping a large
    # JSON representation of the native 3500x7000 arrays.
    click_step = 2
    click_agreement = evaluation["agreement"][::click_step, ::click_step]
    click_strength = evaluation["strength"][::click_step, ::click_step]
    click_dominant = dominant[::click_step, ::click_step]
    click_top = top[::click_step, ::click_step]
    click_second = second[::click_step, ::click_step]
    click_valid = evidence["valid_fraction"][::click_step, ::click_step]
    click_ml = evidence["ml_fraction"][::click_step, ::click_step]
    click_dry = evidence["dry_snow_fraction"][::click_step, ::click_step]
    south, west, north, east = bounds
    click_payload = {
        "status": "ok",
        "purpose": "Compact point diagnostics for the Phase Agreement map.",
        "bounds": [south, west, north, east],
        "shape": [int(click_agreement.shape[0]), int(click_agreement.shape[1])],
        "source_grid": list(evaluation["agreement"].shape),
        "fields": {
            "agreement": click_agreement.astype(int).ravel().tolist(),
            "agreement_strength": np.round(click_strength, 3).ravel().tolist(),
            "dominant_phase": click_dominant.astype(int).ravel().tolist(),
            "top_probability_percent": np.round(click_top, 1).ravel().tolist(),
            "runner_up_probability_percent": np.round(click_second, 1).ravel().tolist(),
            "valid_fraction": np.round(click_valid, 3).ravel().tolist(),
            "melting_layer_fraction": np.round(click_ml, 3).ravel().tolist(),
            "dry_snow_fraction": np.round(click_dry, 3).ravel().tolist(),
        },
    }
    (OUTPUT_DIR / "phase_agreement_click.json").write_text(
        json.dumps(click_payload, separators=(",", ":")),
        encoding="utf-8",
    )

    summary_path = OUTPUT_DIR / "phase_radar_fusion.json"
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    data["overlay_file"] = "phase_radar_fusion_overlay_web.png"
    data["native_overlay_file"] = "phase_radar_fusion_overlay.png"
    data["web_projection"] = "EPSG:3857"
    data["bounds"] = load_lat_lon_bounds(metadata, reflectivity.shape)
    summary_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    del reflectivity, phase_data, placed_phase, dominant, top, second, evidence, evaluation, reduced_rgba, full_rgba, agreement_reduced, agreement_full
    gc.collect()

    print()
    print("=" * 72)
    print("RADAR PHASE EVIDENCE FUSION COMPLETE")
    print(f"  {OUTPUT_DIR / 'phase_radar_fusion.png'}")
    print(f"  {OUTPUT_DIR / 'phase_radar_fusion_overlay.png'}")
    print(f"  {OUTPUT_DIR / 'phase_radar_fusion.json'}")
    print(f"  {OUTPUT_DIR / 'phase_radar_fusion_overlay_web.png'}")
    print(f"  {OUTPUT_DIR / 'phase_radar_fusion_overlay_web.webp'}")
    print(f"  {OUTPUT_DIR / 'phase_agreement.png'}")
    print(f"  {OUTPUT_DIR / 'phase_agreement_overlay.png'}")
    print(f"  {OUTPUT_DIR / 'phase_agreement_overlay_web.png'}")
    print(f"  {OUTPUT_DIR / 'phase_agreement_overlay_web.webp'}")
    print(f"  {OUTPUT_DIR / 'phase_agreement.json'}")
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    finally:
        # Explicitly release Python references before interpreter teardown.
        # This is intentionally here because the GRIB reader uses native
        # eccodes/pygrib allocations.
        gc.collect()
