from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import sys
import gc
import os

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from classifier import CLEAR, ClassificationResult, classify_initial, rain_intensity_dbz  # noqa: E402
from config import DATA_DIR, OUTPUT_DIR, PRODUCTS  # noqa: E402
from download_rap_profile import download_rap_profile, load_rap_profile, sample_profile_to_mrms  # noqa: E402
from download_mrms import download_optional_live_products, download_required_live_products  # noqa: E402
from phase_profile import classify_from_vertical_profile, probabilities_to_phase  # noqa: E402
from read_mrms import get_values, get_valid_time  # noqa: E402
from render import (  # noqa: E402
    reflectivity_to_rgba,
    result_to_phase_rgba,
    result_to_precip_type_rgba,
    save_rgba_png,
    write_metadata,
)

MAIN_VERSION = "9.3-rap-profile-phase-full-conus"
PROFILE_CHUNK_ROWS = 20  # memory-safe phase sampling; still aligned with the 10x diagnostic grid
DIAG_Y_FACTOR = 10
DIAG_X_FACTOR = 10


def _path_for(name: str) -> Path:
    return DATA_DIR / f"MRMS_{PRODUCTS[name]}.latest.grib2"


def load(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return get_values(_path_for(name), product=PRODUCTS[name])


def try_load_optional(name: str, shape: tuple[int, int], default: float) -> np.ndarray:
    path = _path_for(name)
    if not path.exists():
        print(f"  Optional field unavailable: {name} -> default {default}")
        return np.full(shape, default, dtype=np.float32)
    try:
        values, _, _ = load(name)
        if values.shape != shape:
            raise ValueError(f"shape {values.shape} does not match {shape}")
        print(f"  Loaded optional field: {name} {values.shape}")
        return values
    except Exception as exc:
        print(f"  Optional field read failed: {name}: {exc}")
        return np.full(shape, default, dtype=np.float32)


def normalize_longitudes(lons: np.ndarray) -> np.ndarray:
    arr = np.asarray(lons, dtype=np.float64)
    return np.where(arr > 180.0, arr - 360.0, arr)


def grid_bounds(lats: np.ndarray, lons: np.ndarray) -> list[float]:
    lats = np.asarray(lats, dtype=np.float64)
    lons_norm = normalize_longitudes(lons)
    return [float(np.nanmin(lats)), float(np.nanmin(lons_norm)), float(np.nanmax(lats)), float(np.nanmax(lons_norm))]


def get_mrms_valid_time(path: Path) -> str | None:
    """Read MRMS valid time without loading cfgrib/eccodes into the core process."""
    return get_valid_time(path)


def update_metadata(metadata_path: Path, lats: np.ndarray, lons: np.ndarray, mrms_time_utc: str | None) -> None:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update({
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "grid_shape": [int(np.asarray(lats).shape[0]), int(np.asarray(lons).shape[-1] if np.asarray(lons).ndim > 1 else np.asarray(lons).shape[0])],
        "bounds": grid_bounds(lats, lons),
        "bounds_format": ["south", "west", "north", "east"],
        "main_version": MAIN_VERSION,
        "projection": "EPSG:4326-native-latlon",
        "image_origin": "upper",
        "latitude_order": "descending" if np.asarray(lats).ndim == 1 and np.all(np.diff(lats) < 0) else "native",
        "longitude_convention": "-180_to_180",
    })
    if mrms_time_utc:
        metadata["mrms_time_utc"] = mrms_time_utc
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def validate_core_inputs(ref: np.ndarray, lats: np.ndarray, lons: np.ndarray) -> None:
    if ref.ndim != 2:
        raise RuntimeError(f"Reflectivity is not 2-D: {ref.shape}")
    if lats.ndim not in (1, 2) or lons.ndim not in (1, 2):
        raise RuntimeError(f"Unexpected coordinate dimensions: latitude={lats.shape}, longitude={lons.shape}")
    if lats.ndim == 1 and len(lats) != ref.shape[0]:
        raise RuntimeError(f"Latitude length {len(lats)} does not match image rows {ref.shape[0]}")
    if lons.ndim == 1 and len(lons) != ref.shape[1]:
        raise RuntimeError(f"Longitude length {len(lons)} does not match image columns {ref.shape[1]}")


def _profile_phase_result(ref: np.ndarray, lats: np.ndarray, lons: np.ndarray, mrms_time_utc: str) -> tuple[ClassificationResult, dict, dict[str, np.ndarray]]:
    profile_path = download_rap_profile(mrms_time_utc)
    profile = load_rap_profile(profile_path)

    # Refuse to use a stale RAP profile. The profile must be close enough to the
    # MRMS observation time to support a meaningful thermodynamic diagnosis.
    rap_valid = profile.get("valid_time_utc")
    if rap_valid:
        mrms_dt = datetime.fromisoformat(mrms_time_utc.replace("Z", "+00:00"))
        rap_dt = datetime.fromisoformat(rap_valid.replace("Z", "+00:00"))
        age_minutes = abs((mrms_dt - rap_dt).total_seconds()) / 60.0
        if age_minutes > 90.0:
            raise RuntimeError(f"RAP profile is {age_minutes:.0f} minutes from MRMS valid time; refusing stale profile.")
    else:
        age_minutes = None

    phase = np.full(ref.shape, CLEAR, dtype=np.uint8)
    confidence = np.zeros(ref.shape, dtype=np.float32)
    intensity = np.zeros(ref.shape, dtype=np.uint8)

    max_prob = {k: 0.0 for k in ("rain", "snow", "sleet", "freezing_rain")}
    mean_me = []
    mean_re = []
    mean_ice = []

    diag_h = ref.shape[0] // DIAG_Y_FACTOR
    diag_w = ref.shape[1] // DIAG_X_FACTOR
    diag = {
        "rain": np.full((diag_h, diag_w), np.nan, dtype=np.float32),
        "snow": np.full((diag_h, diag_w), np.nan, dtype=np.float32),
        "sleet": np.full((diag_h, diag_w), np.nan, dtype=np.float32),
        "freezing_rain": np.full((diag_h, diag_w), np.nan, dtype=np.float32),
        "melting_energy": np.full((diag_h, diag_w), np.nan, dtype=np.float32),
        "refreezing_energy": np.full((diag_h, diag_w), np.nan, dtype=np.float32),
        "prob_ice": np.full((diag_h, diag_w), np.nan, dtype=np.float32),
    }

    for y0 in range(0, ref.shape[0], PROFILE_CHUNK_ROWS):
        y1 = min(y0 + PROFILE_CHUNK_ROWS, ref.shape[0])
        lat_chunk = lats[y0:y1] if lats.ndim == 1 else lats[y0:y1, :]
        lon_chunk = lons if lons.ndim == 1 else lons[y0:y1, :]
        sampled = sample_profile_to_mrms(profile, lat_chunk, lon_chunk)

        probs = classify_from_vertical_profile(
            pressure_hpa=sampled["pressure_hpa"],
            wetbulb_c=sampled["wetbulb_c"],
            height_m=sampled["height_m"],
            temperature_c=sampled["temperature_c"],
            rh_ice_pct=sampled["rh_ice_pct"],
        )
        ph, conf = probabilities_to_phase(probs)

        # Only classify where the radar actually indicates precipitation.
        precip = np.isfinite(ref[y0:y1]) & (ref[y0:y1] >= 10.0) & sampled["valid"]
        ph[~precip] = CLEAR
        conf[~precip] = 0.0

        phase[y0:y1] = ph
        confidence[y0:y1] = conf
        intensity[y0:y1] = rain_intensity_dbz(ref[y0:y1])
        intensity[y0:y1][~precip] = 0

        # Build a compact 10x-downsampled diagnostic grid. We keep probabilities
        # separate from the categorical mask so the QC panel can show ambiguity.
        if (y1 - y0) % DIAG_Y_FACTOR == 0 and ref.shape[1] % DIAG_X_FACTOR == 0:
            dy0 = y0 // DIAG_Y_FACTOR
            dy1 = y1 // DIAG_Y_FACTOR
            for key in ("rain", "snow", "sleet", "freezing_rain", "melting_energy", "refreezing_energy", "prob_ice"):
                arr = np.asarray(probs[key], dtype=np.float32).copy()
                arr[~precip] = np.nan
                h2 = arr.shape[0] // DIAG_Y_FACTOR
                w2 = arr.shape[1] // DIAG_X_FACTOR
                block = arr.reshape(h2, DIAG_Y_FACTOR, w2, DIAG_X_FACTOR)

                # Do not call np.nanmean() on all-NaN blocks. Some RAP/MRMS
                # sample cells legitimately contain no valid precipitation/profile
                # samples, and np.nanmean() emits a RuntimeWarning for those cells.
                # Sum/count gives the same mean where data exist and leaves truly
                # empty diagnostic cells as NaN without generating warnings.
                valid = np.isfinite(block)
                count = valid.sum(axis=(1, 3))
                total = np.nansum(block, axis=(1, 3))
                block_mean = np.full((h2, w2), np.nan, dtype=np.float32)
                np.divide(
                    total,
                    count,
                    out=block_mean,
                    where=count > 0,
                )
                diag[key][dy0:dy1] = block_mean

        for key in max_prob:
            values = np.asarray(probs[key], dtype=np.float32)
            finite = values[np.isfinite(values)]
            if finite.size:
                max_prob[key] = max(max_prob[key], float(np.max(finite)))

        for values, target in (
            (probs["melting_energy"], mean_me),
            (probs["refreezing_energy"], mean_re),
            (probs["prob_ice"], mean_ice),
        ):
            finite = np.asarray(values, dtype=np.float32)
            finite = finite[np.isfinite(finite)]
            if finite.size:
                target.append(float(np.mean(finite)))

        # Release the large temporary 3-D NumPy arrays before moving to the
        # next chunk. This keeps the full-CONUS phase engine below the runner's
        # memory ceiling and avoids native-library cleanup crashes at shutdown.
        del sampled, probs, ph, conf, precip
        gc.collect()

    diagnostics = {
        "engine": "Modified Bourgouin (Birk et al. 2021)",
        "profile_source": "RAP 13-km pressure-level subset",
        "rap_profile_file": str(profile_path.name),
        "pressure_levels_hpa": [float(x) for x in profile["pressure_hpa"]],
        "profile_model": "RAP 13-km",
        "rap_valid_time_utc": profile.get("valid_time_utc"),
        "rap_mrms_time_offset_minutes": age_minutes,
        "phase_domain": {
            "west": -130.0, "east": -60.0, "south": 20.0, "north": 55.0,
        },
        "max_probabilities_percent": max_prob,
        "mean_melting_energy_jkg": float(np.mean(mean_me)) if mean_me else None,
        "mean_refreezing_energy_jkg": float(np.mean(mean_re)) if mean_re else None,
        "mean_prob_ice_percent": float(np.mean(mean_ice)) if mean_ice else None,
    }
    diagnostics["precip_pixels"] = int(np.count_nonzero(np.isfinite(ref) & (ref >= 10.0)))
    diagnostic_valid = np.isfinite(diag["rain"])
    diagnostics["diagnostic_grid"] = {
        "width": int(diag_w),
        "height": int(diag_h),
        "downsample_factor": 10,
        "valid_cells": int(np.count_nonzero(diagnostic_valid)),
        "total_cells": int(diagnostic_valid.size),
        "valid_fraction": float(np.count_nonzero(diagnostic_valid) / max(diagnostic_valid.size, 1)),
    }

    # Record a warning rather than failing the phase engine.  A live scan can
    # legitimately have no precipitating pixels, while a nonzero precipitation
    # count with an empty diagnostic grid points to a sampling/grid problem.
    if diagnostics["precip_pixels"] > 0 and not np.any(diagnostic_valid):
        diagnostics["diagnostic_warning"] = (
            "No valid phase diagnostic cells were produced despite precipitating "
            "MRMS pixels; check chunk/grid alignment and RAP profile sampling."
        )
        print(f"  WARNING: {diagnostics['diagnostic_warning']}")

    return ClassificationResult(phase=phase, confidence=confidence, intensity=intensity), diagnostics, diag


def main() -> None:
    print("=" * 72)
    print(f"WINTER RADAR CORE — {MAIN_VERSION}")
    print("Radar generation remains isolated from the research phase engine.")
    print("Native MRMS grid; Web Mercator projection is handled downstream.")
    print("=" * 72)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading REQUIRED live MRMS reflectivity...")
    required_status = download_required_live_products()
    if required_status.get("reflectivity") is None:
        raise RuntimeError("Required MRMS reflectivity download failed.")

    ref, lats, lons = load("reflectivity")
    validate_core_inputs(ref, lats, lons)
    lons_norm = normalize_longitudes(lons)

    print(f"Reflectivity shape: {ref.shape}")
    print(f"Latitude range: {float(np.nanmin(lats)):.3f} to {float(np.nanmax(lats)):.3f}")
    print(f"Longitude range: {float(np.nanmin(lons_norm)):.3f} to {float(np.nanmax(lons_norm)):.3f}")

    # Radar is written before any phase processing.
    print("Writing live MRMS reflectivity overlay...")
    save_rgba_png(reflectivity_to_rgba(ref), OUTPUT_DIR / "mrms_current.png")

    print("Downloading optional MRMS phase-support fields...")
    optional_status = download_optional_live_products()
    live_status = {**required_status, **optional_status}
    shape = ref.shape
    precip_flag = try_load_optional("precip_flag", shape, np.nan)
    bb_top = try_load_optional("bb_top", shape, np.nan)
    bb_bottom = try_load_optional("bb_bottom", shape, np.nan)
    rqi = try_load_optional("rqi", shape, 1.0)
    wetbulb = try_load_optional("wetbulb", shape, np.nan)

    metadata_path = OUTPUT_DIR / "mrms_current.json"
    mrms_time_utc = get_mrms_valid_time(_path_for("reflectivity"))

    print("Building scientifically based winter phase mask...")
    phase_status = "fallback_initial"
    phase_error = None
    phase_diagnostics = {}
    phase_probability_data = None
    try:
        if not mrms_time_utc:
            raise RuntimeError("MRMS valid time unavailable; cannot synchronize RAP profile.")
        result, phase_diagnostics, phase_probability_data = _profile_phase_result(ref, lats, lons_norm, mrms_time_utc)
        phase_status = "modified_bourgouin_rap"
    except Exception as exc:
        phase_error = f"{type(exc).__name__}: {exc}"
        print(f"  WARNING: profile phase engine failed; retaining conservative fallback: {phase_error}")
        result = classify_initial(
            reflectivity=ref,
            precip_flag=precip_flag,
            bb_top_m=bb_top,
            bb_bottom_m=bb_bottom,
            wetbulb_c=wetbulb,
            freezing_level_m=None,
            rqi=rqi,
        )
        diag_h = max(1, ref.shape[0] // DIAG_Y_FACTOR)
        diag_w = max(1, ref.shape[1] // DIAG_X_FACTOR)
        phase_probability_data = {
            "rain": np.full((diag_h, diag_w), np.nan, dtype=np.float32),
            "snow": np.full((diag_h, diag_w), np.nan, dtype=np.float32),
            "sleet": np.full((diag_h, diag_w), np.nan, dtype=np.float32),
            "freezing_rain": np.full((diag_h, diag_w), np.nan, dtype=np.float32),
            "melting_energy": np.full((diag_h, diag_w), np.nan, dtype=np.float32),
            "refreezing_energy": np.full((diag_h, diag_w), np.nan, dtype=np.float32),
            "prob_ice": np.full((diag_h, diag_w), np.nan, dtype=np.float32),
        }

    np.savez_compressed(OUTPUT_DIR / "phase_probabilities.npz", **phase_probability_data)

    save_rgba_png(
        result_to_phase_rgba(result),
        OUTPUT_DIR / "winter_phase_mask.png",
    )

    # Screenshot-style all-precipitation display.  This is a visualization
    # of the existing phase solution, not a new classifier.
    save_rgba_png(
        result_to_precip_type_rgba(result, ref),
        OUTPUT_DIR / "winter_precip_type.png",
    )

    write_metadata(result, metadata_path)
    update_metadata(metadata_path, lats, lons_norm, mrms_time_utc)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["phase_status"] = phase_status
    metadata["phase_support_fields"] = {
        "precip_flag": bool(live_status.get("precip_flag")),
        "bb_top": bool(live_status.get("bb_top")),
        "bb_bottom": bool(live_status.get("bb_bottom")),
        "rqi": bool(live_status.get("rqi")),
        "wetbulb": bool(live_status.get("wetbulb")),
    }
    if phase_error:
        metadata["phase_error"] = phase_error
    if phase_diagnostics:
        metadata["phase_diagnostics"] = phase_diagnostics
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    np.save(OUTPUT_DIR / "latitude.npy", np.asarray(lats))
    np.save(OUTPUT_DIR / "longitude.npy", lons_norm)

    expected = [
        OUTPUT_DIR / "mrms_current.png",
        OUTPUT_DIR / "winter_phase_mask.png",
        OUTPUT_DIR / "winter_precip_type.png",
        OUTPUT_DIR / "mrms_current.json",
    ]
    for path in expected:
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Core output missing or empty: {path}")

    print("=" * 72)
    print("CORE MRMS OUTPUTS READY")
    for path in expected:
        print(f"  {path} ({path.stat().st_size:,} bytes)")
    print("=" * 72)


if __name__ == "__main__":
    main()
    # ecCodes/cfgrib can occasionally segfault during interpreter teardown on
    # GitHub-hosted runners after a large full-CONUS run. All required files
    # are closed and written before this point; exiting directly prevents a
    # harmless native finalizer crash from being reported as workflow failure.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
