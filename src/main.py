from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import gc
import json
import sys
import gc

import matplotlib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from classifier import (  # noqa: E402
    ClassificationResult,
    classify_initial,
)
from config import DATA_DIR, OUTPUT_DIR, PRODUCTS  # noqa: E402
from download_mrms import main as download_all  # noqa: E402
from read_mrms import get_values  # noqa: E402
# Cartopy is used only for the final Web-Mercator raster reprojection.
from render import (  # noqa: E402
    reflectivity_to_rgba,
    result_to_phase_rgba,
    save_rgba_png,
    write_metadata,
)



def load(name: str):
    product = PRODUCTS[name]
    return get_values(
        DATA_DIR / f"MRMS_{product}.latest.grib2",
        product=product,
    )


def grid_bounds(lats: np.ndarray, lons: np.ndarray) -> list[float]:
    """Return [south, west, north, east] for the native MRMS grid."""
    lats = np.asarray(lats)
    lons = np.asarray(lons)
    lons_norm = np.where(lons > 180.0, lons - 360.0, lons)
    return [
        float(np.nanmin(lats)),
        float(np.nanmin(lons_norm)),
        float(np.nanmax(lats)),
        float(np.nanmax(lons_norm)),
    ]


def get_mrms_valid_time(path: Path) -> str | None:
    """Read the actual valid timestamp from the reflectivity GRIB."""
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



def update_metadata_bounds(metadata_path: Path, bounds: list[float], mrms_time_utc: str | None = None) -> None:
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
    metadata["projection"] = "EPSG:4326-native-latlon"
    if mrms_time_utc:
        metadata["mrms_time_utc"] = mrms_time_utc

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )



def normalize_longitudes(lons: np.ndarray) -> np.ndarray:
    """Convert MRMS east-positive longitudes to -180..180."""
    arr = np.asarray(lons, dtype=np.float64)
    return np.where(arr > 180.0, arr - 360.0, arr)


def crop_to_regional(
    data: np.ndarray,
    phase: np.ndarray,
    confidence: np.ndarray,
    intensity: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
) -> tuple[np.ndarray, ClassificationResult, np.ndarray, np.ndarray, list[float]]:
    """
    Extract the regional MRMS source window without changing the native
    north-up scan orientation.

    MRMS is a regular 0.01-degree lat/lon grid with latitude decreasing
    from north to south and longitude increasing west to east.
    """
    lat = np.asarray(lats, dtype=np.float64).reshape(-1)
    lon = normalize_longitudes(lons).reshape(-1)

    row_keep = (lat >= MAP_SOUTH) & (lat <= MAP_NORTH)
    col_keep = (lon >= MAP_WEST) & (lon <= MAP_EAST)

    rows = np.where(row_keep)[0]
    cols = np.where(col_keep)[0]

    if rows.size == 0 or cols.size == 0:
        raise RuntimeError("MRMS grid does not overlap regional extent.")

    r0, r1 = rows.min(), rows.max() + 1
    c0, c1 = cols.min(), cols.max() + 1

    ref_crop = np.ascontiguousarray(data[r0:r1, c0:c1])
    phase_crop = np.ascontiguousarray(phase[r0:r1, c0:c1])
    confidence_crop = np.ascontiguousarray(confidence[r0:r1, c0:c1])
    intensity_crop = np.ascontiguousarray(intensity[r0:r1, c0:c1])

    lat_crop = lat[r0:r1].copy()
    lon_crop = lon[c0:c1].copy()

    regional_result = ClassificationResult(
        phase=phase_crop,
        confidence=confidence_crop,
        intensity=intensity_crop,
    )

    return (
        ref_crop,
        regional_result,
        lat_crop,
        lon_crop,
        [MAP_SOUTH, MAP_WEST, MAP_NORTH, MAP_EAST],
    )


def render_projected_rgba(
    rgba: np.ndarray,
    source_lats: np.ndarray,
    source_lons: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    """
    Reproject an RGBA raster from the native MRMS Plate Carrée grid into
    EPSG:3857/Web Mercator.

    This is the key fix for the geographic displacement seen when a
    latitude-linear MRMS raster is stretched directly over a Web Mercator
    Leaflet map.
    """
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    lat = np.asarray(source_lats, dtype=np.float64).reshape(-1)
    lon = np.asarray(source_lons, dtype=np.float64).reshape(-1)

    # MRMS native coordinates are cell centers. Convert them to cell-edge
    # extents so the image is not shifted by half a grid cell.
    dx = float(np.nanmedian(np.diff(lon)))
    dy = float(abs(np.nanmedian(np.diff(lat))))

    west = float(lon[0] - dx / 2.0)
    east = float(lon[-1] + dx / 2.0)
    north = float(lat[0] + dy / 2.0)
    south = float(lat[-1] - dy / 2.0)

    # Keep the rendered image tied to the requested regional map extent.
    west = max(west, MAP_WEST)
    east = min(east, MAP_EAST)
    south = max(south, MAP_SOUTH)
    north = min(north, MAP_NORTH)

    src_crs = ccrs.PlateCarree()
    dst_crs = ccrs.Mercator(
        central_longitude=0.0,
        min_latitude=-80.0,
        max_latitude=80.0,
    )

    # Match figure aspect to the projected geographic extent.
    def merc_y(lat_deg: float) -> float:
        rad = np.deg2rad(np.clip(lat_deg, -85.05112878, 85.05112878))
        return float(
            6378137.0 * np.log(np.tan(np.pi / 4.0 + rad / 2.0))
        )

    x_width = 6378137.0 * np.deg2rad(east - west)
    y_height = merc_y(north) - merc_y(south)
    aspect = x_width / y_height

    width_in = 16.0
    height_in = width_in / aspect

    fig = plt.figure(
        figsize=(width_in, height_in),
        dpi=100,
        facecolor="none",
    )

    ax = fig.add_axes(
        [0.0, 0.0, 1.0, 1.0],
        projection=dst_crs,
    )

    ax.set_extent(
        [MAP_WEST, MAP_EAST, MAP_SOUTH, MAP_NORTH],
        crs=src_crs,
    )
    ax.set_axis_off()

    ax.imshow(
        rgba,
        origin="upper",
        extent=[west, east, south, north],
        transform=src_crs,
        interpolation="nearest",
    )

    fig.savefig(
        output_path,
        dpi=100,
        transparent=True,
        bbox_inches=None,
        pad_inches=0,
    )

    plt.close(fig)

    print(f"Projected {title} written: {output_path}")
    print(
        "  Projection: EPSG:3857 / Web Mercator"
    )
    print(
        f"  Source extent: "
        f"{south:.4f}, {west:.4f}, {north:.4f}, {east:.4f}"
    )


def get_mrms_valid_time(path: Path) -> str | None:
    """Read the actual valid timestamp from the reflectivity GRIB."""
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
        print(
            f"Warning: unable to read MRMS valid time: {exc}"
        )
        return None


def update_metadata(
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
        datetime.now(timezone.utc).isoformat()
    )
    metadata["bounds"] = bounds
    metadata["bounds_format"] = [
        "south",
        "west",
        "north",
        "east",
    ]
    metadata["projection"] = (
        "EPSG:3857-Web-Mercator"
    )
    metadata["main_version"] = (
        "6.0-webmercator-cartopy"
    )

    if mrms_time_utc:
        metadata["mrms_time_utc"] = (
            mrms_time_utc
        )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2
        ),
        encoding="utf-8",
    )


def main() -> None:
    print("=" * 72)
    print(
        "WINTERRADAR MAIN MRMS PROCESSING — "
        "VERSION 6.0-webmercator-cartopy"
    )
    print("=" * 72)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Downloading latest MRMS data...")
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
            f"MRMS valid time: {mrms_time_utc}"
        )

    print("Loading MRMS fields...")

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
        f"  Reflectivity shape: {ref.shape}"
    )
    print(
        "  Native latitude range: "
        f"{float(np.nanmin(lats)):.3f} to "
        f"{float(np.nanmax(lats)):.3f}"
    )

    lons_norm = normalize_longitudes(
        lons
    )

    print(
        "  Native longitude range: "
        f"{float(np.nanmin(lons_norm)):.3f} to "
        f"{float(np.nanmax(lons_norm)):.3f}"
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

    print(
        "Extracting regional MRMS source grid..."
    )

    (
        ref_region,
        result_region,
        lat_region,
        lon_region,
        bounds,
    ) = crop_to_regional(
        ref,
        result.phase,
        result.confidence,
        result.intensity,
        lats,
        lons_norm,
    )

    print(
        f"  Regional grid: "
        f"{ref_region.shape[0]} rows x "
        f"{ref_region.shape[1]} columns"
    )

    print(
        "Rendering radar in Web Mercator..."
    )

    radar_rgba = reflectivity_to_rgba(
        ref_region
    )

    render_projected_rgba(
        radar_rgba,
        lat_region,
        lon_region,
        OUTPUT_DIR / "mrms_current.png",
        "MRMS radar",
    )

    print(
        "Rendering winter mask in Web Mercator..."
    )

    phase_rgba = result_to_phase_rgba(
        result_region
    )

    render_projected_rgba(
        phase_rgba,
        lat_region,
        lon_region,
        OUTPUT_DIR / "winter_phase_mask.png",
        "winter phase mask",
    )

    metadata_path = (
        OUTPUT_DIR
        / "mrms_current.json"
    )

    write_metadata(
        result_region,
        metadata_path,
    )

    update_metadata(
        metadata_path,
        bounds,
        mrms_time_utc,
    )

    # Save the coordinates actually used by the web-map raster.
    np.save(
        OUTPUT_DIR / "latitude.npy",
        lat_region,
    )

    np.save(
        OUTPUT_DIR / "longitude.npy",
        lon_region,
    )

    print("Created:")
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
        ref_region,
        result_region,
        radar_rgba,
        phase_rgba,
        lat_region,
        lon_region,
        lats,
        lons,
        lons_norm,
    )

    gc.collect()


if __name__ == "__main__":
    main()
