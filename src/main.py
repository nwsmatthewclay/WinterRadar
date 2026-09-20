from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import sys

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from classifier import CLEAR, ClassificationResult, classify_initial, rain_intensity_dbz  # noqa: E402
from config import DATA_DIR, OUTPUT_DIR, PRODUCTS  # noqa: E402
from download_rap_profile import download_rap_profile, load_rap_profile, sample_profile_to_mrms  # noqa: E402
from download_mrms import download_optional_live_products, download_required_live_products  # noqa: E402
from phase_profile import classify_from_vertical_profile, probabilities_to_phase  # noqa: E402
from read_mrms import get_values  # noqa: E402
from render import reflectivity_to_rgba, result_to_phase_rgba, save_rgba_png, write_metadata  # noqa: E402

MAIN_VERSION = "9.1-rap-profile-phase"
PROFILE_CHUNK_ROWS = 50
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
    """Read MRMS valid time through cfgrib/xarray; pygrib is not needed."""
    try:
        ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
        try:
            if "valid_time" in ds.coords:
                value = np.asarray(ds.valid_time.values).reshape(-1)[0]
            elif "time" in ds.coords:
                value = np.asarray(ds.time.values).reshape(-1)[0]
            else:
                return None
            dt = value.astype("datetime64[us]").astype(datetime).replace(tzinfo=timezone.utc)
            return dt.isoformat()
        finally:
            ds.close()
    except Exception as exc:
        print(f"  Warning: could not read MRMS valid time: {exc}")
        return None


def update_metadata(metadata_path: Path, lats: np.ndarray, lons: np.ndarray, mrms_time_utc: str | None) -> None:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update({
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
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

    # The RAP subset is intentionally small. Do not run a 21-level calculation
    # over the entire 3500x7000 MRMS CONUS grid: outside the RAP domain every
    # sampled profile point is invalid and all that work is discarded. Restrict
    # the expensive profile calculation to the actual RAP phase-analysis domain.
    if np.asarray(lats).ndim != 1 or np.asarray(lons).ndim != 1:
        raise RuntimeError("Profile-phase optimization expects 1-D MRMS latitude/longitude coordinates.")

    rap_lat = np.asarray(profile["latitude"], dtype=np.float64)
    rap_lon = np.asarray(profile["longitude"], dtype=np.float64)
    south = float(np.nanmin(rap_lat))
    north = float(np.nanmax(rap_lat))
    west = float(np.nanmin(rap_lon))
    east = float(np.nanmax(rap_lon))

    lat_mask = (lats >= south) & (lats <= north)
    lon_mask = (lons >= west) & (lons <= east)
    y_indices = np.flatnonzero(lat_mask)
    x_indices = np.flatnonzero(lon_mask)
    if y_indices.size == 0 or x_indices.size == 0:
        raise RuntimeError("MRMS grid does not intersect the downloaded RAP phase-analysis domain.")

    y0_domain, y1_domain = int(y_indices[0]), int(y_indices[-1]) + 1
    x0_domain, x1_domain = int(x_indices[0]), int(x_indices[-1]) + 1
    ref_domain = ref[y0_domain:y1_domain, x0_domain:x1_domain]
    lats_domain = lats[y0_domain:y1_domain]
    lons_domain = lons[x0_domain:x1_domain]

    phase = np.full(ref.shape, CLEAR, dtype=np.uint8)
    confidence = np.zeros(ref.shape, dtype=np.float32)
    intensity = np.zeros(ref.shape, dtype=np.uint8)

    work_h, work_w = ref_domain.shape
    diag_h = (work_h + DIAG_Y_FACTOR - 1) // DIAG_Y_FACTOR
    diag_w = (work_w + DIAG_X_FACTOR - 1) // DIAG_X_FACTOR
    diag = {
        "rain": np.full((diag_h, diag_w), np.nan, dtype=np.float32),
        "snow": np.full((diag_h, diag_w), np.nan, dtype=np.float32),
        "sleet": np.full((diag_h, diag_w), np.nan, dtype=np.float32),
        "freezing_rain": np.full((diag_h, diag_w), np.nan, dtype=np.float32),
        "melting_energy": np.full((diag_h, diag_w), np.nan, dtype=np.float32),
        "refreezing_energy": np.full((diag_h, diag_w), np.nan, dtype=np.float32),
        "prob_ice": np.full((diag_h, diag_w), np.nan, dtype=np.float32),
    }

    max_prob = {k: 0.0 for k in ("rain", "snow", "sleet", "freezing_rain")}
    mean_me = []
    mean_re = []
    mean_ice = []

    print(
        f"  RAP phase processing domain: rows {y0_domain}:{y1_domain}, "
        f"cols {x0_domain}:{x1_domain} ({work_h}x{work_w})",
        flush=True,
    )

    # Only the RAP-covered portion of MRMS is processed. This cuts the phase
    # workload by more than an order of magnitude relative to the full CONUS
    # grid while leaving the final radar/phase image at its original dimensions.
    for y0 in range(0, work_h, PROFILE_CHUNK_ROWS):
        y1 = min(y0 + PROFILE_CHUNK_ROWS, work_h)
        print(f"  RAP phase chunk: {y1}/{work_h} rows", flush=True)
        lat_chunk = lats_domain[y0:y1]
        sampled = sample_profile_to_mrms(profile, lat_chunk, lons_domain)

        probs = classify_from_vertical_profile(
            pressure_hpa=sampled["pressure_hpa"],
            wetbulb_c=sampled["wetbulb_c"],
            height_m=sampled["height_m"],
            temperature_c=sampled["temperature_c"],
            rh_ice_pct=sampled["rh_ice_pct"],
        )
        ph, conf = probabilities_to_phase(probs)

        precip = np.isfinite(ref_domain[y0:y1]) & (ref_domain[y0:y1] >= 10.0) & sampled["valid"]
        ph[~precip] = CLEAR
        conf[~precip] = 0.0

        phase[y0_domain + y0:y0_domain + y1, x0_domain:x1_domain] = ph
        confidence[y0_domain + y0:y0_domain + y1, x0_domain:x1_domain] = conf
        inten = rain_intensity_dbz(ref_domain[y0:y1])
        inten[~precip] = 0
        intensity[y0_domain + y0:y0_domain + y1, x0_domain:x1_domain] = inten

        # Cheap representative 10x10 sampling for the QC artifact. The final
        # map remains full-resolution; these arrays are diagnostics only.
        for key in ("rain", "snow", "sleet", "freezing_rain", "melting_energy", "refreezing_energy", "prob_ice"):
            arr = np.asarray(probs[key], dtype=np.float32).copy()
            arr[~precip] = np.nan
            sample = arr[::DIAG_Y_FACTOR, ::DIAG_X_FACTOR]
            dy0 = (y0 // DIAG_Y_FACTOR)
            dy1 = min(dy0 + sample.shape[0], diag_h)
            diag[key][dy0:dy1, :sample.shape[1]] = sample[:dy1 - dy0]

        for key in max_prob:
            finite = probs[key][np.isfinite(probs[key])]
            if finite.size:
                max_prob[key] = max(max_prob[key], float(np.max(finite)))
        if np.isfinite(probs["melting_energy"]).any():
            mean_me.append(float(np.nanmean(probs["melting_energy"])))
        if np.isfinite(probs["refreezing_energy"]).any():
            mean_re.append(float(np.nanmean(probs["refreezing_energy"])))
        if np.isfinite(probs["prob_ice"]).any():
            mean_ice.append(float(np.nanmean(probs["prob_ice"])))

    diagnostics = {
        "engine": "Modified Bourgouin (Birk et al. 2021)",
        "profile_source": "RAP 13-km pressure-level subset",
        "rap_profile_file": str(profile_path.name),
        "pressure_levels_hpa": [float(x) for x in profile["pressure_hpa"]],
        "profile_model": "RAP 13-km",
        "rap_valid_time_utc": profile.get("valid_time_utc"),
        "rap_mrms_time_offset_minutes": age_minutes,
        "phase_domain": {
            "west": west, "east": east, "south": south, "north": north,
        },
        "phase_domain_mrms_indices": {
            "row_start": y0_domain, "row_end": y1_domain,
            "col_start": x0_domain, "col_end": x1_domain,
        },
        "phase_domain_shape": {"height": work_h, "width": work_w},
        "max_probabilities_percent": max_prob,
        "mean_melting_energy_jkg": float(np.mean(mean_me)) if mean_me else None,
        "mean_refreezing_energy_jkg": float(np.mean(mean_re)) if mean_re else None,
        "mean_prob_ice_percent": float(np.mean(mean_ice)) if mean_ice else None,
    }
    diagnostics["precip_pixels"] = int(np.count_nonzero(np.isfinite(ref_domain) & (ref_domain >= 10.0)))
    diagnostics["diagnostic_grid"] = {"width": int(diag_w), "height": int(diag_h), "downsample_factor": 10}
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

    save_rgba_png(result_to_phase_rgba(result), OUTPUT_DIR / "winter_phase_mask.png")
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
