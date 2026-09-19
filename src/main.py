from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import gc
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from classifier import classify_initial  # noqa: E402
from config import DATA_DIR, OUTPUT_DIR, PRODUCTS  # noqa: E402
from download_mrms import main as download_all  # noqa: E402
from read_mrms import get_values  # noqa: E402
from render import (  # noqa: E402
    reflectivity_to_rgba,
    result_to_phase_rgba,
    save_rgba_png,
    write_metadata,
)

# Explicit marker so the Actions log proves which main.py is running.
MAIN_VERSION = "3.0-btv-crop"

# Regional map extent for the BTV CWA / surrounding area.
MAP_SOUTH = 41.50
MAP_WEST = -76.50
MAP_NORTH = 45.60
MAP_EAST = -69.50


def load(name: str):
    product = PRODUCTS[name]
    return get_values(
        DATA_DIR / f"MRMS_{product}.latest.grib2",
        product=product,
    )


def normalize_longitudes(lons: np.ndarray) -> np.ndarray:
    """Convert longitude coordinates to the conventional -180..180 range."""
    arr = np.asarray(lons, dtype=np.float64).copy()
    arr = np.where(arr > 180.0, arr - 360.0, arr)
    return arr


def normalize_orientation(
    arrays: list[np.ndarray],
    lats: np.ndarray,
    lons: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Ensure row 0 is north and column 0 is west."""
    lats_work = np.asarray(lats).copy()
    lons_work = normalize_longitudes(lons)
    result_arrays = [np.asarray(a) for a in arrays]

    # MRMS latitude/longitude axes are normally 1-D.
    if lats_work.ndim == 1:
        lat_step = float(np.nanmedian(np.diff(lats_work)))
        if np.isfinite(lat_step) and lat_step > 0:
            result_arrays = [np.flipud(a) for a in result_arrays]
            lats_work = np.flipud(lats_work)

    if lons_work.ndim == 1:
        lon_step = float(np.nanmedian(np.diff(lons_work)))
        if np.isfinite(lon_step) and lon_step < 0:
            result_arrays = [np.fliplr(a) for a in result_arrays]
            lons_work = np.flip(lons_work)

    # Fallback for 2-D coordinate arrays.
    elif lats_work.ndim == 2 and lons_work.ndim == 2:
        lat_step = float(np.nanmedian(lats_work[1:, :] - lats_work[:-1, :]))
        if np.isfinite(lat_step) and lat_step > 0:
            result_arrays = [np.flipud(a) for a in result_arrays]
            lats_work = np.flipud(lats_work)
            lons_work = np.flipud(lons_work)

        lon_step = float(np.nanmedian(lons_work[:, 1:] - lons_work[:, :-1]))
        if np.isfinite(lon_step) and lon_step < 0:
            result_arrays = [np.fliplr(a) for a in result_arrays]
            lats_work = np.fliplr(lats_work)
            lons_work = np.fliplr(lons_work)

    return result_arrays, lats_work, lons_work


def _nearest_index(axis: np.ndarray, value: float) -> int:
    """Index of the coordinate nearest to value."""
    axis = np.asarray(axis)
    idx = int(np.nanargmin(np.abs(axis - value)))
    return idx


def crop_to_btv_region(
    arrays: list[np.ndarray],
    lats: np.ndarray,
    lons: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """
    Crop by nearest coordinate indices rather than boolean masks.

    This avoids the false 'no overlap' failure caused by small coordinate
    convention differences and works with either increasing or decreasing
    1-D axes after normalization.
    """
    lats_arr = np.asarray(lats)
    lons_arr = normalize_longitudes(lons)

    print("Coordinate check before BTV crop:")
    print(
        f"  Latitude shape: {lats_arr.shape}, "
        f"range: {float(np.nanmin(lats_arr)):.3f} to {float(np.nanmax(lats_arr)):.3f}"
    )
    print(
        f"  Longitude shape: {lons_arr.shape}, "
        f"range: {float(np.nanmin(lons_arr)):.3f} to {float(np.nanmax(lons_arr)):.3f}"
    )
    if lats_arr.ndim == 1:
        print(f"  Latitude samples: {lats_arr[0]:.3f}, {lats_arr[-1]:.3f}")
    if lons_arr.ndim == 1:
        print(f"  Longitude samples: {lons_arr[0]:.3f}, {lons_arr[-1]:.3f}")

    # 1-D axes: this is the normal MRMS case.
    if lats_arr.ndim == 1 and lons_arr.ndim == 1:
        lat_min = float(np.nanmin(lats_arr))
        lat_max = float(np.nanmax(lats_arr))
        lon_min = float(np.nanmin(lons_arr))
        lon_max = float(np.nanmax(lons_arr))

        if not (lat_min <= MAP_NORTH and lat_max >= MAP_SOUTH):
            raise RuntimeError(
                f"MRMS latitude range {lat_min:.3f}..{lat_max:.3f} "
                f"does not include BTV extent {MAP_SOUTH:.2f}..{MAP_NORTH:.2f}."
            )
        if not (lon_min <= MAP_EAST and lon_max >= MAP_WEST):
            raise RuntimeError(
                f"MRMS longitude range {lon_min:.3f}..{lon_max:.3f} "
                f"does not include BTV extent {MAP_WEST:.2f}..{MAP_EAST:.2f}."
            )

        r_a = _nearest_index(lats_arr, MAP_SOUTH)
        r_b = _nearest_index(lats_arr, MAP_NORTH)
        c_a = _nearest_index(lons_arr, MAP_WEST)
        c_b = _nearest_index(lons_arr, MAP_EAST)

        r0, r1 = sorted((r_a, r_b))
        c0, c1 = sorted((c_a, c_b))

        # One grid-cell of padding keeps the requested extent fully inside.
        r0 = max(0, r0 - 1)
        r1 = min(lats_arr.size - 1, r1 + 1)
        c0 = max(0, c0 - 1)
        c1 = min(lons_arr.size - 1, c1 + 1)

        # End indices are exclusive.
        r1 += 1
        c1 += 1

        cropped = [np.ascontiguousarray(a[r0:r1, c0:c1]) for a in arrays]

        print(f"  Crop rows: {r0}:{r1}  ({r1-r0})")
        print(f"  Crop cols: {c0}:{c1}  ({c1-c0})")

        return cropped, lats_arr[r0:r1].copy(), lons_arr[c0:c1].copy()

    # 2-D fallback.
    inside = (
        np.isfinite(lats_arr)
        & np.isfinite(lons_arr)
        & (lats_arr >= MAP_SOUTH)
        & (lats_arr <= MAP_NORTH)
        & (lons_arr >= MAP_WEST)
        & (lons_arr <= MAP_EAST)
    )

    rows = np.where(np.any(inside, axis=1))[0]
    cols = np.where(np.any(inside, axis=0))[0]

    if rows.size == 0 or cols.size == 0:
        raise RuntimeError("MRMS 2-D coordinate grid does not overlap the BTV map extent.")

    r0 = max(0, int(rows[0]) - 1)
    r1 = min(lats_arr.shape[0], int(rows[-1]) + 2)
    c0 = max(0, int(cols[0]) - 1)
    c1 = min(lons_arr.shape[1], int(cols[-1]) + 2)

    cropped = [np.ascontiguousarray(a[r0:r1, c0:c1]) for a in arrays]
    return (
        cropped,
        np.ascontiguousarray(lats_arr[r0:r1, c0:c1]),
        np.ascontiguousarray(lons_arr[r0:r1, c0:c1]),
    )


def grid_bounds(lats: np.ndarray, lons: np.ndarray) -> list[float]:
    return [
        float(np.nanmin(lats)),
        float(np.nanmin(lons)),
        float(np.nanmax(lats)),
        float(np.nanmax(lons)),
    ]


def write_map_metadata(metadata_path: Path, bounds: list[float]) -> None:
    metadata = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}

    metadata["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["bounds"] = bounds
    metadata["bounds_format"] = ["south", "west", "north", "east"]
    metadata["main_version"] = MAIN_VERSION

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    print("=" * 72)
    print(f"WINTERRADAR MAIN MRMS PROCESSING — VERSION {MAIN_VERSION}")
    print("=" * 72)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading latest MRMS data...")
    download_all()

    print("Loading MRMS fields...")
    ref, lats, lons = load("reflectivity")
    pflag, _, _ = load("precip_flag")
    bb_top, _, _ = load("bb_top")
    bb_bottom, _, _ = load("bb_bottom")
    rqi, _, _ = load("rqi")
    wetbulb, _, _ = load("wetbulb")
    frz, _, _ = load("freezing_level")

    # Downloaded for diagnostics/science work; not needed to render the two map overlays.
    _precip_rate, _, _ = load("precip_rate")
    del _precip_rate

    print("Classifying winter precipitation...")
    classification = classify_initial(
        reflectivity=ref,
        precip_flag=pflag,
        bb_top_m=bb_top,
        bb_bottom_m=bb_bottom,
        wetbulb_c=wetbulb,
        freezing_level_m=frz,
        rqi=rqi,
    )

    normalized, lats, lons = normalize_orientation(
        [
            ref,
            classification.phase,
            classification.confidence,
            classification.intensity,
        ],
        lats,
        lons,
    )

    ref = normalized[0]
    classification.phase = normalized[1]
    classification.confidence = normalized[2]
    classification.intensity = normalized[3]

    print("Cropping products to the BTV regional map area...")
    cropped, lats, lons = crop_to_btv_region(
        [
            ref,
            classification.phase,
            classification.confidence,
            classification.intensity,
        ],
        lats,
        lons,
    )

    ref = cropped[0]
    classification.phase = cropped[1]
    classification.confidence = cropped[2]
    classification.intensity = cropped[3]

    bounds = grid_bounds(lats, lons)
    print("Final map bounds:", bounds)

    print("Writing MRMS radar overlay...")
    radar_rgba = reflectivity_to_rgba(ref)
    save_rgba_png(radar_rgba, OUTPUT_DIR / "mrms_current.png")

    print("Writing winter phase overlay...")
    phase_rgba = result_to_phase_rgba(classification)
    save_rgba_png(phase_rgba, OUTPUT_DIR / "winter_phase_mask.png")

    metadata_path = OUTPUT_DIR / "mrms_current.json"
    write_metadata(classification, metadata_path)
    write_map_metadata(metadata_path, bounds)

    np.save(OUTPUT_DIR / "latitude.npy", lats)
    np.save(OUTPUT_DIR / "longitude.npy", lons)

    # Release large native arrays before process shutdown.
    del ref, pflag, bb_top, bb_bottom, rqi, wetbulb, frz, classification
    gc.collect()

    print("Created:")
    print("  outputs/mrms_current.png")
    print("  outputs/winter_phase_mask.png")
    print("  outputs/mrms_current.json")
    print("MAIN MRMS PROCESSING COMPLETE")


if __name__ == "__main__":
    main()
