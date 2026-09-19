from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from classifier import (
    CLEAR,
    ClassificationResult,
    classify_initial,
)  # noqa: E402
from config import DATA_DIR, OUTPUT_DIR, PRODUCTS  # noqa: E402
from download_mrms import (
    download_optional_live_products,
    download_required_live_products,
)  # noqa: E402
from read_mrms import get_values, get_valid_time_utc  # noqa: E402
from render import reflectivity_to_rgba, result_to_phase_rgba, save_rgba_png, write_metadata  # noqa: E402

MAIN_VERSION = "8.0-stable-live-core"


def _path_for(name: str) -> Path:
    return DATA_DIR / f"MRMS_{PRODUCTS[name]}.latest.grib2"


def load(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return get_values(_path_for(name), product=PRODUCTS[name])


def try_load_optional(
    name: str,
    shape: tuple[int, int],
    default: float,
) -> np.ndarray:
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
    return [
        float(np.nanmin(lats)),
        float(np.nanmin(lons_norm)),
        float(np.nanmax(lats)),
        float(np.nanmax(lons_norm)),
    ]


def update_metadata(
    metadata_path: Path,
    lats: np.ndarray,
    lons: np.ndarray,
    mrms_time_utc: str | None,
) -> None:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "bounds": grid_bounds(lats, lons),
            "bounds_format": ["south", "west", "north", "east"],
            "main_version": MAIN_VERSION,
            "projection": "EPSG:4326-native-latlon",
            "image_origin": "upper",
            "latitude_order": "descending" if np.asarray(lats).ndim == 1 and np.all(np.diff(lats) < 0) else "native",
            "longitude_convention": "-180_to_180",
        }
    )
    if mrms_time_utc:
        metadata["mrms_time_utc"] = mrms_time_utc

    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def validate_core_inputs(ref: np.ndarray, lats: np.ndarray, lons: np.ndarray) -> None:
    if ref.ndim != 2:
        raise RuntimeError(f"Reflectivity is not 2-D: {ref.shape}")

    if lats.ndim not in (1, 2) or lons.ndim not in (1, 2):
        raise RuntimeError(
            f"Unexpected coordinate dimensions: latitude={lats.shape}, longitude={lons.shape}"
        )

    if lats.ndim == 1 and len(lats) != ref.shape[0]:
        raise RuntimeError(
            f"Latitude length {len(lats)} does not match image rows {ref.shape[0]}"
        )
    if lons.ndim == 1 and len(lons) != ref.shape[1]:
        raise RuntimeError(
            f"Longitude length {len(lons)} does not match image columns {ref.shape[1]}"
        )


def main() -> None:
    print("=" * 72)
    print(f"WINTER RADAR CORE — {MAIN_VERSION}")
    print("Radar generation is isolated from optional diagnostics.")
    print("No crop. No reprojection. Native MRMS latitude/longitude grid.")
    print("=" * 72)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading REQUIRED live MRMS reflectivity...")
    required_status = download_required_live_products()

    if required_status.get("reflectivity") is None:
        raise RuntimeError("Required MRMS reflectivity download failed.")

    ref, lats, lons = load("reflectivity")
    validate_core_inputs(ref, lats, lons)

    print(f"Reflectivity shape: {ref.shape}")
    print(
        f"Latitude range: {float(np.nanmin(lats)):.3f} to {float(np.nanmax(lats)):.3f}"
    )
    lons_norm = normalize_longitudes(lons)
    print(
        f"Longitude range: {float(np.nanmin(lons_norm)):.3f} to {float(np.nanmax(lons_norm)):.3f}"
    )

    # Write the radar image immediately after reflectivity is successfully
    # decoded. This ensures phase-processing problems cannot suppress the
    # core MRMS radar product.
    print("Writing live MRMS reflectivity overlay...")
    save_rgba_png(
        reflectivity_to_rgba(ref),
        OUTPUT_DIR / "mrms_current.png",
    )

    print("Downloading optional phase-support MRMS fields...")
    optional_status = download_optional_live_products()
    live_status = {**required_status, **optional_status}

    shape = ref.shape
    precip_flag = try_load_optional("precip_flag", shape, np.nan)
    bb_top = try_load_optional("bb_top", shape, np.nan)
    bb_bottom = try_load_optional("bb_bottom", shape, np.nan)
    rqi = try_load_optional("rqi", shape, 1.0)
    wetbulb = try_load_optional("wetbulb", shape, np.nan)

    print("Classifying winter precipitation...")
    phase_error = None
    try:
        result = classify_initial(
            reflectivity=ref,
            precip_flag=precip_flag,
            bb_top_m=bb_top,
            bb_bottom_m=bb_bottom,
            wetbulb_c=wetbulb,
            freezing_level_m=None,
            rqi=rqi,
        )
        phase_status = "ok"
    except Exception as exc:
        # Never throw away a valid live radar image because the research/phase
        # classifier encountered a bad optional field or native-library issue.
        phase_error = f"{type(exc).__name__}: {exc}"
        print(f"  WARNING: phase classifier failed: {phase_error}")
        result = ClassificationResult(
            phase=np.full(shape, CLEAR, dtype=np.uint8),
            confidence=np.zeros(shape, dtype=np.float32),
            intensity=np.zeros(shape, dtype=np.uint8),
        )
        phase_status = "fallback_transparent"

    print("Writing winter phase overlay...")
    save_rgba_png(
        result_to_phase_rgba(result),
        OUTPUT_DIR / "winter_phase_mask.png",
    )

    metadata_path = OUTPUT_DIR / "mrms_current.json"
    write_metadata(result, metadata_path)

    # Read the timestamp through cfgrib/ecCodes instead of pygrib. This is
    # intentionally best-effort: timestamp extraction can never invalidate a
    # radar image that has already been successfully written.
    try:
        mrms_time_utc = get_valid_time_utc(_path_for("reflectivity"))
    except Exception as exc:
        print(f"  Warning: could not read MRMS valid time: {exc}")
        mrms_time_utc = None
    if mrms_time_utc:
        print(f"MRMS valid time: {mrms_time_utc}")

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
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    np.save(OUTPUT_DIR / "latitude.npy", np.asarray(lats))
    np.save(OUTPUT_DIR / "longitude.npy", lons_norm)

    # Final in-process QC. Fail loudly here rather than allowing a broken
    # artifact to reach Pages.
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

    # Do not force garbage collection here.  The live core uses native GRIB
    # libraries underneath cfgrib; explicit shutdown/GC at this point is not
    # necessary and can trigger third-party native finalizers after all output
    # files are already valid.  Let normal process teardown reclaim memory.


if __name__ == "__main__":
    main()
