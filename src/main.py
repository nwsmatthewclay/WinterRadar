from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from classifier import classify_initial
from config import DATA_DIR, OUTPUT_DIR, PRODUCTS
from download_mrms import main as download_all
from read_mrms import get_values
from render import result_to_rgba, save_rgba_png, write_metadata


def load(name: str):
    return get_values(DATA_DIR / f"MRMS_{PRODUCTS[name]}.latest.grib2")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    download_all()

    ref, lats, lons = load("reflectivity")
    pflag, _, _ = load("precip_flag")
    bb_top, _, _ = load("bb_top")
    bb_bottom, _, _ = load("bb_bottom")
    rqi, _, _ = load("rqi")
    wetbulb, _, _ = load("wetbulb")
    frz, _, _ = load("freezing_level")

    # MRMS fields are expected on the same grid. If a future product differs,
    # the next iteration will add a common-grid resampler here.
    result = classify_initial(
        reflectivity=ref,
        precip_flag=pflag,
        bb_top_m=bb_top,
        bb_bottom_m=bb_bottom,
        wetbulb_c=wetbulb,
        freezing_level_m=frz,
        rqi=rqi,
    )

    rgba = result_to_rgba(result)
    save_rgba_png(rgba, OUTPUT_DIR / "mrms_current.png")
    write_metadata(result, OUTPUT_DIR / "mrms_current.json")

    np.save(OUTPUT_DIR / "latitude.npy", lats)
    np.save(OUTPUT_DIR / "longitude.npy", lons)
    print("Wrote outputs/mrms_current.png and outputs/mrms_current.json")


if __name__ == "__main__":
    main()
