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

# This is intentionally based on the last known-good MRMS main.py.
# The failing BTV crop logic has been removed.
MAIN_VERSION = "4.0-known-good-no-crop"


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


def update_metadata_bounds(metadata_path: Path, bounds: list[float]) -> None:
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



def get_mrms_valid_time(path: Path) -> str | None:
    """Read the actual MRMS valid time from the first GRIB message."""
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

def main() -> None:
    print("=" * 72)
    print(f"WINTERRADAR MAIN MRMS PROCESSING — VERSION {MAIN_VERSION}")
    print("Using the last known-good native MRMS grid; NO spatial crop.")
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

    print(f"  Reflectivity shape: {ref.shape}")
    print(f"  Latitude range: {float(np.nanmin(lats)):.3f} to {float(np.nanmax(lats)):.3f}")
    lons_norm = np.where(np.asarray(lons) > 180.0, np.asarray(lons) - 360.0, np.asarray(lons))
    print(f"  Longitude range: {float(np.nanmin(lons_norm)):.3f} to {float(np.nanmax(lons_norm)):.3f}")

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

    print("Writing MRMS radar overlay...")
    radar_rgba = reflectivity_to_rgba(ref)
    save_rgba_png(radar_rgba, OUTPUT_DIR / "mrms_current.png")

    print("Writing winter phase overlay...")
    phase_rgba = result_to_phase_rgba(result)
    save_rgba_png(phase_rgba, OUTPUT_DIR / "winter_phase_mask.png")

    metadata_path = OUTPUT_DIR / "mrms_current.json"
    write_metadata(result, metadata_path)
    update_metadata_bounds(metadata_path, grid_bounds(lats, lons))

    # Store the actual MRMS valid time for the dashboard.
    mrms_time = get_mrms_valid_time(
        DATA_DIR / f"MRMS_{PRODUCTS['reflectivity']}.latest.grib2"
    )
    if mrms_time:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["mrms_time_utc"] = mrms_time
        metadata_path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

    # Keep these for the existing site/UI tooling.
    np.save(OUTPUT_DIR / "latitude.npy", np.asarray(lats))
    np.save(OUTPUT_DIR / "longitude.npy", lons_norm)

    print("Created:")
    print("  outputs/mrms_current.png")
    print("  outputs/winter_phase_mask.png")
    print("  outputs/mrms_current.json")
    print("MAIN MRMS PROCESSING COMPLETE")

    # Release large arrays before native-library shutdown.
    del ref, pflag, bb_top, bb_bottom, rqi, wetbulb, frz, result
    gc.collect()


if __name__ == "__main__":
    main()
