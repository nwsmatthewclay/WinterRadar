from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent),
)

from classifier import classify_initial

from config import (
    DATA_DIR,
    OUTPUT_DIR,
    PRODUCTS,
)

from download_mrms import main as download_all

from read_mrms import get_values

from render import (
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


def _normalize_grid_orientation(
    arrays: list[np.ndarray],
    lats: np.ndarray,
    lons: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """
    Make sure the first image row represents the north side of the
    grid and the first image column represents the west side.

    Leaflet image overlays assume the top-left of the image corresponds
    to the northwest corner of the geographic bounds.
    """

    lats_work = np.asarray(lats)
    lons_work = np.asarray(lons)

    normalized = [np.asarray(a) for a in arrays]

    # Latitude direction.
    if lats_work.ndim == 2 and lats_work.shape[0] > 1:
        lat_step = np.nanmedian(
            lats_work[1:, :] - lats_work[:-1, :]
        )

    elif lats_work.ndim == 1 and lats_work.size > 1:
        lat_step = np.nanmedian(
            np.diff(lats_work)
        )

    else:
        lat_step = np.nan

    # If latitude increases downward, flip north/south.
    if np.isfinite(lat_step) and lat_step > 0:
        normalized = [
            np.flipud(a)
            for a in normalized
        ]
        lats_work = np.flipud(lats_work)
        lons_work = np.flipud(lons_work)

    # Longitude direction.
    if lons_work.ndim == 2 and lons_work.shape[1] > 1:
        lon_step = np.nanmedian(
            lons_work[:, 1:] - lons_work[:, :-1]
        )

    elif lons_work.ndim == 1 and lons_work.size > 1:
        lon_step = np.nanmedian(
            np.diff(lons_work)
        )

    else:
        lon_step = np.nan

    # If longitude decreases to the right, flip west/east.
    if np.isfinite(lon_step) and lon_step < 0:
        normalized = [
            np.fliplr(a)
            for a in normalized
        ]
        lats_work = np.fliplr(lats_work)
        lons_work = np.fliplr(lons_work)

    return normalized, lats_work, lons_work


def _grid_bounds(
    lats: np.ndarray,
    lons: np.ndarray,
) -> list[float]:
    """
    Return [south, west, north, east] in decimal degrees.
    """

    lat_valid = np.asarray(lats, dtype=np.float64)
    lon_valid = np.asarray(lons, dtype=np.float64)

    south = float(
        np.nanmin(lat_valid)
    )
    north = float(
        np.nanmax(lat_valid)
    )
    west = float(
        np.nanmin(lon_valid)
    )
    east = float(
        np.nanmax(lon_valid)
    )

    return [
        south,
        west,
        north,
        east,
    ]


def _update_metadata_with_map_info(
    metadata_path: Path,
    bounds: list[float],
) -> None:
    """
    Add geographic placement information to the existing MRMS JSON.
    This avoids requiring a separate workflow output/copy step.
    """

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

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Downloading latest MRMS data...")

    download_all()

    print("Loading MRMS fields...")

    # --------------------------------------------------------------
    # Core fields
    # --------------------------------------------------------------

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

    precip_rate, _, _ = load(
        "precip_rate"
    )

    del precip_rate

    print(
        "Running winter precipitation classifier..."
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

    # --------------------------------------------------------------
    # Normalize geographic orientation before rendering.
    # The same flip is applied to reflectivity and phase.
    # --------------------------------------------------------------

    normalized, lats, lons = _normalize_grid_orientation(
        [
            ref,
            result.phase,
            result.confidence,
            result.intensity,
        ],
        lats,
        lons,
    )

    ref = normalized[0]

    result.phase = normalized[1]
    result.confidence = normalized[2]
    result.intensity = normalized[3]

    # --------------------------------------------------------------
    # MAIN MAP LAYER
    #
    # Transparent MRMS reflectivity image.
    # Leaflet places this image geographically over the basemap.
    # --------------------------------------------------------------

    radar_rgba = reflectivity_to_rgba(
        ref
    )

    save_rgba_png(
        radar_rgba,
        OUTPUT_DIR / "mrms_current.png",
    )

    # --------------------------------------------------------------
    # WINTER PHASE LAYER
    #
    # Separate transparent overlay so the web map can turn it on/off
    # independently of the radar.
    # --------------------------------------------------------------

    phase_rgba = result_to_phase_rgba(
        result
    )

    save_rgba_png(
        phase_rgba,
        OUTPUT_DIR / "winter_phase_mask.png",
    )

    # --------------------------------------------------------------
    # Metadata
    # --------------------------------------------------------------

    metadata_path = (
        OUTPUT_DIR /
        "mrms_current.json"
    )

    write_metadata(
        result,
        metadata_path,
    )

    bounds = _grid_bounds(
        lats,
        lons,
    )

    _update_metadata_with_map_info(
        metadata_path,
        bounds,
    )

    # --------------------------------------------------------------
    # Keep coordinate arrays available for research/debugging.
    # --------------------------------------------------------------

    np.save(
        OUTPUT_DIR / "latitude.npy",
        lats,
    )

    np.save(
        OUTPUT_DIR / "longitude.npy",
        lons,
    )

    print(
        "Wrote:"
        "\n  outputs/mrms_current.png"
        "\n  outputs/winter_phase_mask.png"
        "\n  outputs/mrms_current.json"
    )

    print(
        "Map bounds [south, west, north, east]:",
        bounds,
    )


if __name__ == "__main__":
    main()
