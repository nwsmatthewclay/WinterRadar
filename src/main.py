from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parent
    ),
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
    result_to_rgba,
    save_rgba_png,
    write_metadata,
)


def load(name: str):

    product = PRODUCTS[name]

    path = (
        DATA_DIR
        / f"MRMS_{product}.latest.grib2"
    )

    print(
        f"Loading {name}: {path}"
    )

    return get_values(
        path,
        product=product,
    )


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Downloading latest MRMS fields..."
    )

    download_all()

    print(
        "Loading MRMS fields..."
    )

    # ---------------------------------------------------------------
    # Core radar / precipitation fields
    # ---------------------------------------------------------------

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

    freezing_level, _, _ = load(
        "freezing_level"
    )

    # Precipitation rate is retained in the pipeline for future
    # accumulation/intensity work.
    precip_rate, _, _ = load(
        "precip_rate"
    )

    del precip_rate

    # ---------------------------------------------------------------
    # Classification
    # ---------------------------------------------------------------

    print(
        "Classifying precipitation phase..."
    )

    result = classify_initial(
        reflectivity=ref,
        precip_flag=pflag,
        bb_top_m=bb_top,
        bb_bottom_m=bb_bottom,
        wetbulb_c=wetbulb,
        freezing_level_m=freezing_level,
        rqi=rqi,
    )

    # ---------------------------------------------------------------
    # Winter-only transparent overlay
    # ---------------------------------------------------------------

    rgba = result_to_rgba(
        result
    )

    current_png = (
        OUTPUT_DIR
        / "mrms_current.png"
    )

    winter_png = (
        OUTPUT_DIR
        / "winter_phase_mask.png"
    )

    metadata_json = (
        OUTPUT_DIR
        / "mrms_current.json"
    )

    save_rgba_png(
        rgba,
        current_png,
    )

    # Separate copy with an explicit winter-mask filename.
    save_rgba_png(
        rgba,
        winter_png,
    )

    write_metadata(
        result,
        metadata_json,
    )

    # ---------------------------------------------------------------
    # Save grid coordinates
    # ---------------------------------------------------------------

    np.save(
        OUTPUT_DIR / "latitude.npy",
        lats,
    )

    np.save(
        OUTPUT_DIR / "longitude.npy",
        lons,
    )

    # ---------------------------------------------------------------
    # Console summary
    # ---------------------------------------------------------------

    unique, counts = np.unique(
        result.phase,
        return_counts=True,
    )

    names = {
        0: "CLEAR",
        1: "RAIN",
        2: "SNOW",
        3: "SLEET",
        4: "FZRA",
        5: "MIXED",
        9: "UNKNOWN",
    }

    print("")
    print(
        "========================================"
    )
    print(
        " MRMS WINTER PHASE SUMMARY"
    )
    print(
        "========================================"
    )

    for phase, count in zip(
        unique,
        counts,
    ):

        name = names.get(
            int(phase),
            f"PHASE_{int(phase)}",
        )

        pct = (
            100.0
            * float(count)
            / float(result.phase.size)
        )

        print(
            f"{name:8s}: "
            f"{int(count):10d} "
            f"({pct:6.2f}%)"
        )

    print(
        "========================================"
    )

    print(
        f"Wrote {current_png}"
    )

    print(
        f"Wrote {winter_png}"
    )

    print(
        f"Wrote {metadata_json}"
    )


if __name__ == "__main__":
    main()
