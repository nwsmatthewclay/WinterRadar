
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

# ----------------------------------------------------------------------
# LIVE MRMS MAP
# ----------------------------------------------------------------------
# The MRMS 2-D products are a regular latitude/longitude grid.
# Leaflet/Web Mercator does NOT space pixels linearly in latitude.
#
# Instead of stretching the native image over geographic bounds, we
# explicitly warp the image rows into a Web-Mercator-linear Y grid.
#
# This preserves the working MRMS product and fixes the north/south
# displacement without requiring Cartopy, GDAL, or a second projection
# library.
# ----------------------------------------------------------------------

MAIN_VERSION = "7.0-webmercator-warp"

MAP_SOUTH = 40.95
MAP_WEST = -77.50
MAP_NORTH = 46.15
MAP_EAST = -68.40

MAP_WIDTH = 1600
EARTH_RADIUS_M = 6378137.0


def load(name: str):
    product = PRODUCTS[name]

    return get_values(
        DATA_DIR / f"MRMS_{product}.latest.grib2",
        product=product,
    )


def normalize_longitudes(lons: np.ndarray) -> np.ndarray:
    arr = np.asarray(
        lons,
        dtype=np.float64,
    )

    return np.where(
        arr > 180.0,
        arr - 360.0,
        arr,
    )


def get_mrms_valid_time(path: Path) -> str | None:
    """Read the actual MRMS valid time from the GRIB message."""

    try:
        import pygrib

        grbs = pygrib.open(
            str(path)
        )

        try:
            msg = grbs.message(1)

            dt = getattr(
                msg,
                "validDate",
                None,
            )

            if dt is None:
                return None

            if dt.tzinfo is None:
                dt = dt.replace(
                    tzinfo=timezone.utc
                )
            else:
                dt = dt.astimezone(
                    timezone.utc
                )

            return dt.isoformat()

        finally:
            grbs.close()

    except Exception as exc:
        print(
            "Warning: unable to read MRMS valid time:",
            exc,
        )

        return None


def lat_to_mercator_y(
    lat_deg: np.ndarray | float,
) -> np.ndarray:
    lat_rad = np.deg2rad(
        np.clip(
            lat_deg,
            -85.05112878,
            85.05112878,
        )
    )

    return (
        EARTH_RADIUS_M
        * np.log(
            np.tan(
                np.pi / 4.0
                + lat_rad / 2.0
            )
        )
    )


def mercator_y_to_lat(
    y_m: np.ndarray,
) -> np.ndarray:
    return np.rad2deg(
        2.0
        * np.arctan(
            np.exp(
                y_m / EARTH_RADIUS_M
            )
        )
        - np.pi / 2.0
    )


def nearest_indices_descending(
    axis: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """
    Nearest-neighbor lookup for a monotonically descending axis.
    """

    axis = np.asarray(
        axis,
        dtype=np.float64,
    )

    targets = np.asarray(
        targets,
        dtype=np.float64,
    )

    # Reverse to ascending for searchsorted.
    rev = axis[::-1]

    positions = np.searchsorted(
        rev,
        targets,
        side="left",
    )

    positions = np.clip(
        positions,
        0,
        rev.size - 1,
    )

    previous = np.maximum(
        positions - 1,
        0,
    )

    choose_previous = (
        np.abs(
            targets
            - rev[previous]
        )
        <=
        np.abs(
            targets
            - rev[positions]
        )
    )

    chosen = np.where(
        choose_previous,
        previous,
        positions,
    )

    return (
        axis.size
        - 1
        - chosen
    ).astype(
        np.int64
    )


def regional_warp_to_webmercator(
    ref: np.ndarray,
    phase: np.ndarray,
    confidence: np.ndarray,
    intensity: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
) -> tuple[
    np.ndarray,
    "ClassificationResult",
    np.ndarray,
    np.ndarray,
    list[float],
]:
    """
    Extract the regional MRMS grid and warp it so image rows are linear
    in Web-Mercator Y. Longitude is already linear in Web Mercator X.
    """

    from classifier import ClassificationResult

    lat_axis = np.asarray(
        lats,
        dtype=np.float64,
    ).reshape(-1)

    lon_axis = normalize_longitudes(
        lons
    ).reshape(-1)

    if ref.ndim != 2:
        raise RuntimeError(
            f"Expected 2-D reflectivity; got {ref.shape}"
        )

    if (
        lat_axis.size != ref.shape[0]
        or lon_axis.size != ref.shape[1]
    ):
        raise RuntimeError(
            "MRMS coordinate lengths do not match the raster."
        )

    # MRMS is normally north-to-south in row order.
    lat_step = float(
        np.nanmedian(
            np.diff(lat_axis)
        )
    )

    if lat_step > 0:
        raise RuntimeError(
            "Unexpected MRMS latitude orientation: "
            "expected a descending latitude axis."
        )

    lon_step = float(
        np.nanmedian(
            np.diff(lon_axis)
        )
    )

    if lon_step < 0:
        raise RuntimeError(
            "Unexpected MRMS longitude orientation: "
            "expected an ascending longitude axis."
        )

    # Select only the regional source window.
    row_keep = (
        (lat_axis >= MAP_SOUTH)
        & (lat_axis <= MAP_NORTH)
    )

    col_keep = (
        (lon_axis >= MAP_WEST)
        & (lon_axis <= MAP_EAST)
    )

    rows = np.where(
        row_keep
    )[0]

    cols = np.where(
        col_keep
    )[0]

    if rows.size == 0:
        raise RuntimeError(
            "No MRMS rows overlap the regional map."
        )

    if cols.size == 0:
        raise RuntimeError(
            "No MRMS columns overlap the regional map."
        )

    r0 = int(rows.min())
    r1 = int(rows.max()) + 1

    c0 = int(cols.min())
    c1 = int(cols.max()) + 1

    ref_src = np.ascontiguousarray(
        ref[r0:r1, c0:c1]
    )

    phase_src = np.ascontiguousarray(
        phase[r0:r1, c0:c1]
    )

    confidence_src = np.ascontiguousarray(
        confidence[r0:r1, c0:c1]
    )

    intensity_src = np.ascontiguousarray(
        intensity[r0:r1, c0:c1]
    )

    lat_src = lat_axis[r0:r1]
    lon_src = lon_axis[c0:c1]

    # Calculate a target image height that preserves the geographic
    # aspect ratio in Web Mercator.
    x_w = float(
        EARTH_RADIUS_M
        * np.deg2rad(
            MAP_WEST
        )
    )

    x_e = float(
        EARTH_RADIUS_M
        * np.deg2rad(
            MAP_EAST
        )
    )

    y_s = float(
        lat_to_mercator_y(
            MAP_SOUTH
        )
    )

    y_n = float(
        lat_to_mercator_y(
            MAP_NORTH
        )
    )

    projected_width = (
        x_e - x_w
    )

    projected_height = (
        y_n - y_s
    )

    aspect = (
        projected_width
        / projected_height
    )

    map_height = max(
        700,
        int(
            round(
                MAP_WIDTH
                / aspect
            )
        ),
    )

    # Image row 0 = north. Therefore Y centers go from north to south.
    y_centers = np.linspace(
        y_n,
        y_s,
        map_height,
        dtype=np.float64,
    )

    target_lat = mercator_y_to_lat(
        y_centers
    )

    # Web Mercator X is linear in longitude, so evenly spaced lon
    # centers are sufficient horizontally.
    x_centers = np.linspace(
        x_w,
        x_e,
        MAP_WIDTH,
        dtype=np.float64,
    )

    target_lon = np.rad2deg(
        x_centers
        / EARTH_RADIUS_M
    )

    row_idx = nearest_indices_descending(
        lat_src,
        target_lat,
    )

    col_idx = np.searchsorted(
        lon_src,
        target_lon,
        side="left",
    )

    col_idx = np.clip(
        col_idx,
        0,
        lon_src.size - 1,
    )

    previous = np.maximum(
        col_idx - 1,
        0,
    )

    choose_previous = (
        np.abs(
            target_lon
            - lon_src[previous]
        )
        <=
        np.abs(
            target_lon
            - lon_src[col_idx]
        )
    )

    col_idx = np.where(
        choose_previous,
        previous,
        col_idx,
    ).astype(
        np.int64
    )

    indexer = np.ix_(
        row_idx,
        col_idx,
    )

    ref_map = ref_src[indexer]

    phase_map = phase_src[indexer]

    confidence_map = confidence_src[
        indexer
    ]

    intensity_map = intensity_src[
        indexer
    ]

    result_map = ClassificationResult(
        phase=phase_map,
        confidence=confidence_map,
        intensity=intensity_map,
    )

    print(
        "Web-Mercator raster warp:"
    )

    print(
        f"  Source rows: {r0}:{r1} "
        f"({r1 - r0})"
    )

    print(
        f"  Source cols: {c0}:{c1} "
        f"({c1 - c0})"
    )

    print(
        f"  Output image: "
        f"{MAP_WIDTH} x {map_height}"
    )

    print(
        "  Output bounds: "
        f"{MAP_SOUTH}, {MAP_WEST}, "
        f"{MAP_NORTH}, {MAP_EAST}"
    )

    return (
        ref_map,
        result_map,
        target_lat,
        target_lon,
        [
            MAP_SOUTH,
            MAP_WEST,
            MAP_NORTH,
            MAP_EAST,
        ],
    )


def write_map_metadata(
    metadata_path: Path,
    bounds: list[float],
    mrms_time_utc: str | None,
) -> None:

    metadata = {}

    if metadata_path.exists():
        try:
            metadata = json.loads(
                metadata_path.read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            metadata = {}

    metadata["generated_at_utc"] = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    metadata["bounds"] = bounds

    metadata["bounds_format"] = [
        "south",
        "west",
        "north",
        "east",
    ]

    metadata["projection"] = (
        "EPSG:3857-Web-Mercator-warp"
    )

    metadata["main_version"] = (
        MAIN_VERSION
    )

    if mrms_time_utc:
        metadata["mrms_time_utc"] = (
            mrms_time_utc
        )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:

    print("=" * 72)

    print(
        "WINTERRADAR MAIN MRMS PROCESSING — "
        f"VERSION {MAIN_VERSION}"
    )

    print("=" * 72)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Downloading latest MRMS data..."
    )

    download_all()

    ref_path = (
        DATA_DIR
        / f"MRMS_{PRODUCTS['reflectivity']}.latest.grib2"
    )

    mrms_time_utc = get_mrms_valid_time(
        ref_path
    )

    if mrms_time_utc:
        print(
            "MRMS valid time:",
            mrms_time_utc,
        )

    print(
        "Loading MRMS fields..."
    )

    ref, lats, lons = load(
        "reflectivity"
    )

    pflag, _, _ = load(
        "precip_flag"
    )

    bb_top, _, _ = load(
        "bb_top"
    )

    bb_bottom, _, _ = load(
        "bb_bottom"
    )

    rqi, _, _ = load(
        "rqi"
    )

    wetbulb, _, _ = load(
        "wetbulb"
    )

    frz, _, _ = load(
        "freezing_level"
    )

    print(
        f"  Reflectivity shape: "
        f"{ref.shape}"
    )

    print(
        "  Native latitude range:",
        f"{float(np.nanmin(lats)):.3f} to "
        f"{float(np.nanmax(lats)):.3f}",
    )

    lons_norm = normalize_longitudes(
        lons
    )

    print(
        "  Native longitude range:",
        f"{float(np.nanmin(lons_norm)):.3f} to "
        f"{float(np.nanmax(lons_norm)):.3f}",
    )

    print(
        "Classifying winter precipitation..."
    )

    result = classify_initial(
        reflectivity=ref,
        precip_flag=pflag,
        bb_top_m=bb_top,
        bb_bottom_m=bb_bottom,
        wetbulb_c=wetbulb,
        freezing_level_m=frz,
        rqi=rqi,
    )

    (
        ref_map,
        result_map,
        target_lat,
        target_lon,
        bounds,
    ) = regional_warp_to_webmercator(
        ref,
        result.phase,
        result.confidence,
        result.intensity,
        lats,
        lons_norm,
    )

    print(
        "Writing MRMS radar overlay..."
    )

    radar_rgba = reflectivity_to_rgba(
        ref_map
    )

    save_rgba_png(
        radar_rgba,
        OUTPUT_DIR
        / "mrms_current.png",
    )

    print(
        "Writing winter phase overlay..."
    )

    phase_rgba = result_to_phase_rgba(
        result_map
    )

    save_rgba_png(
        phase_rgba,
        OUTPUT_DIR
        / "winter_phase_mask.png",
    )

    metadata_path = (
        OUTPUT_DIR
        / "mrms_current.json"
    )

    write_metadata(
        result_map,
        metadata_path,
    )

    write_map_metadata(
        metadata_path,
        bounds,
        mrms_time_utc,
    )

    # Save the actual map-grid coordinate centers.
    np.save(
        OUTPUT_DIR
        / "latitude.npy",
        target_lat,
    )

    np.save(
        OUTPUT_DIR
        / "longitude.npy",
        target_lon,
    )

    print(
        "Created:"
    )

    print(
        "  outputs/mrms_current.png"
    )

    print(
        "  outputs/winter_phase_mask.png"
    )

    print(
        "  outputs/mrms_current.json"
    )

    print(
        "MAIN MRMS PROCESSING COMPLETE"
    )

    del (
        ref,
        pflag,
        bb_top,
        bb_bottom,
        rqi,
        wetbulb,
        frz,
        result,
        ref_map,
        result_map,
        radar_rgba,
        phase_rgba,
        target_lat,
        target_lon,
        lats,
        lons,
        lons_norm,
    )

    gc.collect()


if __name__ == "__main__":
    main()
