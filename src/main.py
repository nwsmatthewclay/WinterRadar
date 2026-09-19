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


# ---------------------------------------------------------------
# BTV regional map extent.
#
# This keeps the browser image manageable and focused on the area
# used by WFO Burlington. The map can still zoom to towns/cities.
# ---------------------------------------------------------------

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


def normalize_orientation(
    arrays: list[np.ndarray],
    lats: np.ndarray,
    lons: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """
    Ensure image row 0 is north and image column 0 is west.
    """

    lats_work = np.asarray(lats)
    lons_work = np.asarray(lons)

    result_arrays = [
        np.asarray(a)
        for a in arrays
    ]

    if lats_work.ndim == 2:

        lat_step = np.nanmedian(
            lats_work[1:, :] -
            lats_work[:-1, :]
        )

    else:

        lat_step = np.nanmedian(
            np.diff(lats_work)
        )

    if (
        np.isfinite(lat_step) and
        lat_step > 0
    ):

        result_arrays = [
            np.flipud(a)
            for a in result_arrays
        ]

        lats_work = np.flipud(
            lats_work
        )

        lons_work = np.flipud(
            lons_work
        )

    if lons_work.ndim == 2:

        lon_step = np.nanmedian(
            lons_work[:, 1:] -
            lons_work[:, :-1]
        )

    else:

        lon_step = np.nanmedian(
            np.diff(lons_work)
        )

    if (
        np.isfinite(lon_step) and
        lon_step < 0
    ):

        result_arrays = [
            np.fliplr(a)
            for a in result_arrays
        ]

        lats_work = np.fliplr(
            lats_work
        )

        lons_work = np.fliplr(
            lons_work
        )

    return (
        result_arrays,
        lats_work,
        lons_work,
    )


def crop_to_btv_region(
    arrays: list[np.ndarray],
    lats: np.ndarray,
    lons: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:

    lats_arr = np.asarray(lats)
    lons_arr = np.asarray(lons)

    if (
        lats_arr.ndim == 1 and
        lons_arr.ndim == 1
    ):

        row_keep = (
            (lats_arr >= MAP_SOUTH) &
            (lats_arr <= MAP_NORTH)
        )

        col_keep = (
            (lons_arr >= MAP_WEST) &
            (lons_arr <= MAP_EAST)
        )

        if (
            not np.any(row_keep) or
            not np.any(col_keep)
        ):
            raise RuntimeError(
                "MRMS grid does not overlap the BTV map extent."
            )

        rows = np.where(row_keep)[0]
        cols = np.where(col_keep)[0]

        r0, r1 = rows[0], rows[-1] + 1
        c0, c1 = cols[0], cols[-1] + 1

        cropped = [
            a[r0:r1, c0:c1]
            for a in arrays
        ]

        return (
            cropped,
            lats_arr[r0:r1],
            lons_arr[c0:c1],
        )

    inside = (
        np.isfinite(lats_arr) &
        np.isfinite(lons_arr) &
        (lats_arr >= MAP_SOUTH) &
        (lats_arr <= MAP_NORTH) &
        (lons_arr >= MAP_WEST) &
        (lons_arr <= MAP_EAST)
    )

    rows = np.where(
        np.any(inside, axis=1)
    )[0]

    cols = np.where(
        np.any(inside, axis=0)
    )[0]

    if (
        rows.size == 0 or
        cols.size == 0
    ):
        raise RuntimeError(
            "MRMS grid does not overlap the BTV map extent."
        )

    r0, r1 = rows[0], rows[-1] + 1
    c0, c1 = cols[0], cols[-1] + 1

    cropped = [
        a[r0:r1, c0:c1]
        for a in arrays
    ]

    return (
        cropped,
        lats_arr[r0:r1, c0:c1],
        lons_arr[r0:r1, c0:c1],
    )


def grid_bounds(
    lats: np.ndarray,
    lons: np.ndarray,
) -> list[float]:

    return [
        float(np.nanmin(lats)),
        float(np.nanmin(lons)),
        float(np.nanmax(lats)),
        float(np.nanmax(lons)),
    ]


def write_map_metadata(
    metadata_path: Path,
    bounds: list[float],
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

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2
        ),
        encoding="utf-8",
    )


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Downloading latest MRMS data..."
    )

    download_all()

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

    precip_rate, _, _ = load(
        "precip_rate"
    )

    del precip_rate

    print(
        "Classifying winter precipitation..."
    )

    classification = classify_initial(
        reflectivity=ref,
        precip_flag=pflag,
        bb_top_m=bb_top,
        bb_bottom_m=bb_bottom,
        wetbulb_c=wetbulb,
        freezing_level_m=frz,
        rqi=rqi,
    )

    (
        normalized,
        lats,
        lons,
    ) = normalize_orientation(
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

    print(
        "Cropping products to the BTV regional map area..."
    )

    (
        cropped,
        lats,
        lons,
    ) = crop_to_btv_region(
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

    bounds = grid_bounds(
        lats,
        lons,
    )

    print(
        "Map bounds:",
        bounds,
    )

    # ---------------------------------------------------------------
    # MRMS RADAR OVERLAY
    # ---------------------------------------------------------------

    radar_rgba = reflectivity_to_rgba(
        ref
    )

    save_rgba_png(
        radar_rgba,
        OUTPUT_DIR /
        "mrms_current.png",
    )

    # ---------------------------------------------------------------
    # WINTER PHASE OVERLAY
    #
    # This file is ALWAYS written, even when every pixel is
    # transparent. This prevents stale/missing artifact behavior.
    # ---------------------------------------------------------------

    phase_rgba = result_to_phase_rgba(
        classification
    )

    save_rgba_png(
        phase_rgba,
        OUTPUT_DIR /
        "winter_phase_mask.png",
    )

    # ---------------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------------

    metadata_path = (
        OUTPUT_DIR /
        "mrms_current.json"
    )

    write_metadata(
        classification,
        metadata_path,
    )

    write_map_metadata(
        metadata_path,
        bounds,
    )

    # ---------------------------------------------------------------
    # Coordinates for research / debugging
    # ---------------------------------------------------------------

    np.save(
        OUTPUT_DIR /
        "latitude.npy",
        lats,
    )

    np.save(
        OUTPUT_DIR /
        "longitude.npy",
        lons,
    )

    print(
        "Created:"
        "\n  outputs/mrms_current.png"
        "\n  outputs/winter_phase_mask.png"
        "\n  outputs/mrms_current.json"
    )


if __name__ == "__main__":
    main()
