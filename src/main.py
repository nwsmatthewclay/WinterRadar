from __future__ import annotations

from pathlib import Path
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
    result_to_rgba,
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


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Downloading latest MRMS data...")

    download_all()

    print("Loading MRMS fields...")

    # --------------------------------------------------------------
    # Core MRMS fields
    # --------------------------------------------------------------

    ref, lats, lons = load("reflectivity")

    pflag, _, _ = load("precip_flag")

    bb_top, _, _ = load("bb_top")

    bb_bottom, _, _ = load("bb_bottom")

    rqi, _, _ = load("rqi")

    wetbulb, _, _ = load("wetbulb")

    frz, _, _ = load("freezing_level")

    # Loaded for availability / diagnostics.
    precip_rate, _, _ = load("precip_rate")

    del precip_rate

    print("Running winter precipitation classifier...")

    # --------------------------------------------------------------
    # Phase classification
    # --------------------------------------------------------------

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
    # MAIN PRODUCT
    #
    # MRMS reflectivity underneath the winter precipitation mask.
    #
    # Rain and no precipitation remain unmasked.
    # --------------------------------------------------------------

    rgba = result_to_rgba(
        result,
        reflectivity=ref,
    )

    save_rgba_png(
        rgba,
        OUTPUT_DIR / "mrms_current.png",
    )

    # --------------------------------------------------------------
    # PURE WINTER PHASE MASK
    #
    # Useful for research / validation.
    # --------------------------------------------------------------

    phase_rgba = result_to_phase_rgba(
        result,
    )

    save_rgba_png(
        phase_rgba,
        OUTPUT_DIR / "winter_phase_mask.png",
    )

    # --------------------------------------------------------------
    # Metadata
    # --------------------------------------------------------------

    write_metadata(
        result,
        OUTPUT_DIR / "mrms_current.json",
    )

    # --------------------------------------------------------------
    # Save coordinates
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


if __name__ == "__main__":
    main()
