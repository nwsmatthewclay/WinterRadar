from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import copy
import gc
import json
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from classifier import ClassificationResult, classify_initial  # noqa: E402
from config import DATA_DIR, OUTPUT_DIR, PRODUCTS  # noqa: E402
from download_mrms import main as download_all  # noqa: E402
from read_mrms import get_values  # noqa: E402
from render import (  # noqa: E402
    reflectivity_to_rgba,
    result_to_phase_rgba,
    save_rgba_png,
    write_metadata,
)

# ----------------------------------------------------------------------
# WinterRadar MRMS map rendering
# ----------------------------------------------------------------------
# The MRMS 3500x7000 product is a regular latitude/longitude grid.
# The web map, however, is Leaflet/Web Mercator.  Stretching the raw
# lat/lon image over a geographic rectangle creates a latitude-dependent
# positional error.  We therefore resample the MRMS data onto a Web
# Mercator image for the regional map before handing it to Leaflet.
#
# This replaces the old destructive BTV crop with a robust regional
# extraction + Web Mercator resampling. The live MRMS path remains a
# single main step and the research diagnostics remain separate.
# ----------------------------------------------------------------------

MAIN_VERSION = "5.0-webmercator-regional"

# Same broad regional area used by the dashboard's Regional button.
MAP_SOUTH = 40.95
MAP_WEST = -77.50
MAP_NORTH = 46.15
MAP_EAST = -68.40

# Output resolution. ~3.5 km pixels is plenty for the web display and
# keeps the PNG small enough for GitHub Pages.
MAP_WIDTH = 1600

EARTH_RADIUS_M = 6378137.0


def load(name: str):
    product = PRODUCTS[name]
    return get_values(
        DATA_DIR / f"MRMS_{product}.latest.grib2",
        product=product,
    )


def normalize_longitudes(lons: np.ndarray) -> np.ndarray:
    arr = np.asarray(lons, dtype=np.float64).copy()
    # MRMS commonly stores CONUS longitudes in 0..360 east-positive form.
    arr = np.where(arr > 180.0, arr - 360.0, arr)
    return arr


def normalize_source_orientation(
    arrays: list[np.ndarray],
    lats: np.ndarray,
    lons: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Make source arrays north-up / west-left and coordinates monotonic."""
    lat_axis = np.asarray(lats, dtype=np.float64).reshape(-1)
    lon_axis = normalize_longitudes(lons).reshape(-1)
    out = [np.asarray(a) for a in arrays]

    if out[0].ndim != 2:
        raise RuntimeError(f"Expected 2-D MRMS raster, got shape {out[0].shape}")

    if lat_axis.size != out[0].shape[0]:
        raise RuntimeError(
            f"Latitude axis length {lat_axis.size} does not match raster rows {out[0].shape[0]}"
        )
    if lon_axis.size != out[0].shape[1]:
        raise RuntimeError(
            f"Longitude axis length {lon_axis.size} does not match raster columns {out[0].shape[1]}"
        )

    # After this, row 0 = north and last row = south.
    lat_step = float(np.nanmedian(np.diff(lat_axis)))
    if not np.isfinite(lat_step):
        raise RuntimeError("Unable to determine MRMS latitude-axis direction.")
    if lat_step > 0:
        out = [np.flipud(a) for a in out]
        lat_axis = lat_axis[::-1].copy()

    # After this, col 0 = west and last col = east.
    lon_step = float(np.nanmedian(np.diff(lon_axis)))
    if not np.isfinite(lon_step):
        raise RuntimeError("Unable to determine MRMS longitude-axis direction.")
    if lon_step < 0:
        out = [np.fliplr(a) for a in out]
        lon_axis = lon_axis[::-1].copy()

    return out, lat_axis, lon_axis


def nearest_indices_monotonic(axis: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Vectorized nearest-neighbor indices for a monotonic 1-D axis."""
    axis = np.asarray(axis, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)

    increasing = axis[0] < axis[-1]
    work = axis if increasing else axis[::-1]

    pos = np.searchsorted(work, targets, side="left")
    pos = np.clip(pos, 0, work.size - 1)

    prev = np.maximum(pos - 1, 0)
    nextv = pos

    choose_prev = np.abs(targets - work[prev]) <= np.abs(targets - work[nextv])
    chosen = np.where(choose_prev, prev, nextv)

    if increasing:
        return chosen.astype(np.int64)
    return (work.size - 1 - chosen).astype(np.int64)


def lon_to_mercator_x(lon_deg: np.ndarray) -> np.ndarray:
    return EARTH_RADIUS_M * np.deg2rad(lon_deg)


def lat_to_mercator_y(lat_deg: np.ndarray) -> np.ndarray:
    lat_rad = np.deg2rad(np.clip(lat_deg, -85.05112878, 85.05112878))
    return EARTH_RADIUS_M * np.log(np.tan(np.pi / 4.0 + lat_rad / 2.0))


def mercator_y_to_lat(y_m: np.ndarray) -> np.ndarray:
    return np.rad2deg(2.0 * np.arctan(np.exp(y_m / EARTH_RADIUS_M)) - np.pi / 2.0)


def regional_webmercator_resample(
    ref: np.ndarray,
    phase: np.ndarray,
    confidence: np.ndarray,
    intensity: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
) -> tuple[np.ndarray, ClassificationResult, list[float], np.ndarray, np.ndarray]:
    """
    Extract the regional source window and nearest-neighbor resample it onto
    an image grid that is linear in Web Mercator coordinates.
    """
    arrays, lat_axis, lon_axis = normalize_source_orientation(
        [ref, phase, confidence, intensity],
        lats,
        lons,
    )
    ref_src, phase_src, conf_src, intensity_src = arrays

    # Small source padding protects the target region from edge/center effects.
    lat_keep = (lat_axis >= MAP_SOUTH - 0.10) & (lat_axis <= MAP_NORTH + 0.10)
    lon_keep = (lon_axis >= MAP_WEST - 0.10) & (lon_axis <= MAP_EAST + 0.10)

    if not np.any(lat_keep):
        raise RuntimeError("MRMS latitude axis does not overlap the regional map extent.")
    if not np.any(lon_keep):
        raise RuntimeError("MRMS longitude axis does not overlap the regional map extent.")

    r0 = int(np.where(lat_keep)[0].min())
    r1 = int(np.where(lat_keep)[0].max()) + 1
    c0 = int(np.where(lon_keep)[0].min())
    c1 = int(np.where(lon_keep)[0].max()) + 1

    ref_src = np.ascontiguousarray(ref_src[r0:r1, c0:c1])
    phase_src = np.ascontiguousarray(phase_src[r0:r1, c0:c1])
    conf_src = np.ascontiguousarray(conf_src[r0:r1, c0:c1])
    intensity_src = np.ascontiguousarray(intensity_src[r0:r1, c0:c1])
    lat_src = lat_axis[r0:r1].copy()
    lon_src = lon_axis[c0:c1].copy()

    # Web Mercator extent of the requested geographic map bounds.
    x_w = float(lon_to_mercator_x(np.array([MAP_WEST]))[0])
    x_e = float(lon_to_mercator_x(np.array([MAP_EAST]))[0])
    y_s = float(lat_to_mercator_y(np.array([MAP_SOUTH]))[0])
    y_n = float(lat_to_mercator_y(np.array([MAP_NORTH]))[0])

    aspect = (x_e - x_w) / (y_n - y_s)
    map_height = max(700, int(round(MAP_WIDTH / aspect)))

    # Pixel-center coordinates in Web Mercator.
    x_edges = np.linspace(x_w, x_e, MAP_WIDTH + 1, dtype=np.float64)
    y_edges = np.linspace(y_s, y_n, map_height + 1, dtype=np.float64)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    # Image row 0 must be north.
    target_lon = np.rad2deg(x_centers / EARTH_RADIUS_M)
    target_lat = mercator_y_to_lat(y_centers[::-1])

    row_idx = nearest_indices_monotonic(lat_src, target_lat)
    col_idx = nearest_indices_monotonic(lon_src, target_lon)

    ref_map = ref_src[np.ix_(row_idx, col_idx)]
    phase_map = phase_src[np.ix_(row_idx, col_idx)]
    conf_map = conf_src[np.ix_(row_idx, col_idx)]
    intensity_map = intensity_src[np.ix_(row_idx, col_idx)]

    projected_result = ClassificationResult(
        phase=phase_map,
        confidence=conf_map,
        intensity=intensity_map,
    )

    bounds = [MAP_SOUTH, MAP_WEST, MAP_NORTH, MAP_EAST]

    print("Web Mercator map projection:")
    print(f"  Source crop rows: {r0}:{r1}  ({r1-r0})")
    print(f"  Source crop cols: {c0}:{c1}  ({c1-c0})")
    print(f"  Output image: {MAP_WIDTH} x {map_height}")
    print(f"  Geographic bounds: {bounds}")

    return ref_map, projected_result, bounds, target_lat, target_lon


def get_mrms_valid_time(path: Path) -> str | None:
    """Read the actual valid timestamp from the GRIB message."""
    try:
        import pygrib

        grbs = pygrib.open(str(path))
        try:
            msg = grbs.message(1)
            dt = getattr(msg, "validDate", None)
            if dt is None:
                return None
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt.isoformat()
        finally:
            grbs.close()
    except Exception as exc:
        print(f"Warning: unable to read MRMS valid time: {exc}")
        return None


def write_map_metadata(
    metadata_path: Path,
    bounds: list[float],
    mrms_time_utc: str | None,
) -> None:
    metadata = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}

    metadata["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["bounds"] = bounds
    metadata["bounds_format"] = ["south", "west", "north", "east"]
    metadata["projection"] = "EPSG:3857-Web-Mercator"
    metadata["main_version"] = MAIN_VERSION

    if mrms_time_utc:
        metadata["mrms_time_utc"] = mrms_time_utc

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

    ref_path = DATA_DIR / f"MRMS_{PRODUCTS['reflectivity']}.latest.grib2"
    mrms_time_utc = get_mrms_valid_time(ref_path)
    if mrms_time_utc:
        print(f"MRMS valid time: {mrms_time_utc}")

    print("Loading MRMS fields...")
    ref, lats, lons = load("reflectivity")
    pflag, _, _ = load("precip_flag")
    bb_top, _, _ = load("bb_top")
    bb_bottom, _, _ = load("bb_bottom")
    rqi, _, _ = load("rqi")
    wetbulb, _, _ = load("wetbulb")
    frz, _, _ = load("freezing_level")

    print(f"  Reflectivity shape: {ref.shape}")
    print(f"  Native latitude range: {float(np.nanmin(lats)):.3f} to {float(np.nanmax(lats)):.3f}")
    lons_norm = normalize_longitudes(lons)
    print(f"  Native longitude range: {float(np.nanmin(lons_norm)):.3f} to {float(np.nanmax(lons_norm)):.3f}")

    print("Classifying winter precipitation...")
    result = classify_initial(
        reflectivity=ref,
        precip_flag=pflag,
        bb_top_m=bb_top,
        bb_bottom_m=bb_bottom,
        wetbulb_c=wetbulb,
        freezing_level_m=frz,
        rqi=rqi,
    )

    # Release the auxiliary full-domain fields before the regional resample.
    del pflag, bb_top, bb_bottom, rqi, wetbulb, frz
    gc.collect()

    print("Reprojecting MRMS to Web Mercator for the regional map...")
    ref_map, result_map, bounds, target_lat, target_lon = regional_webmercator_resample(
        ref,
        result.phase,
        result.confidence,
        result.intensity,
        lats,
        lons_norm,
    )

    print("Writing MRMS radar overlay...")
    radar_rgba = reflectivity_to_rgba(ref_map)
    save_rgba_png(radar_rgba, OUTPUT_DIR / "mrms_current.png")

    print("Writing winter phase overlay...")
    phase_rgba = result_to_phase_rgba(result_map)
    save_rgba_png(phase_rgba, OUTPUT_DIR / "winter_phase_mask.png")

    metadata_path = OUTPUT_DIR / "mrms_current.json"
    write_metadata(result_map, metadata_path)
    write_map_metadata(metadata_path, bounds, mrms_time_utc)

    # Keep map-grid coordinates available for debugging/research.
    np.save(OUTPUT_DIR / "latitude.npy", target_lat)
    np.save(OUTPUT_DIR / "longitude.npy", target_lon)

    print("Created:")
    print("  outputs/mrms_current.png")
    print("  outputs/winter_phase_mask.png")
    print("  outputs/mrms_current.json")
    print("MAIN MRMS PROCESSING COMPLETE")

    del ref, result, ref_map, result_map
    del target_lat, target_lon, lats, lons
    gc.collect()


if __name__ == "__main__":
    main()
