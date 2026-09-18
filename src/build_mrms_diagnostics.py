from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PRODUCTS  # noqa: E402
from read_mrms import get_values  # noqa: E402


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def load(name: str):
    """Load one MRMS field using the configured product name."""
    product = PRODUCTS[name]
    path = DATA_DIR / f"MRMS_{product}.latest.grib2"

    if not path.exists():
        raise FileNotFoundError(f"Missing MRMS field: {path}")

    return get_values(path)


def clean_field(data: np.ndarray, kind: str) -> np.ndarray:
    """Convert MRMS sentinel values to NaN."""
    arr = np.asarray(data, dtype=np.float32).copy()

    if kind in {
        "reflectivity",
        "rhohv",
        "zdr",
        "wetbulb",
        "surface_temp",
    }:
        arr[arr <= -900] = np.nan

    elif kind in {
        "bb_top",
        "bb_bottom",
        "rqi",
        "freezing_level",
        "precip_flag",
    }:
        arr[arr <= -2] = np.nan

    return arr


def downsample(arr: np.ndarray, max_dimension: int = 1400) -> np.ndarray:
    """
    Reduce the field for the diagnostic image.

    We retain the original data for the future classifier; this is only
    for plotting.
    """
    step = max(
        1,
        int(np.ceil(max(arr.shape) / max_dimension)),
    )

    return arr[::step, ::step]


def coordinate_for_plot(lats, lons, arr):
    """Create plotting extent/origin for 1-D MRMS coordinates."""
    lats = np.asarray(lats)
    lons = np.asarray(lons)

    lat_min = float(np.nanmin(lats))
    lat_max = float(np.nanmax(lats))
    lon_min = float(np.nanmin(lons))
    lon_max = float(np.nanmax(lons))

    if lats[0] < lats[-1]:
        origin = "lower"
    else:
        origin = "upper"

    return (
        [lon_min, lon_max, lat_min, lat_max],
        origin,
    )


def add_panel(
    ax,
    data,
    lats,
    lons,
    title,
    cmap,
    vmin=None,
    vmax=None,
    units="",
):
    """Draw one diagnostic panel."""
    plotted = downsample(data)

    extent, origin = coordinate_for_plot(lats, lons, plotted)

    masked = np.ma.masked_invalid(plotted)

    image = ax.imshow(
        masked,
        extent=extent,
        origin=origin,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect="auto",
    )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.15)

    cbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    if units:
        cbar.set_label(units)

    return image


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("")
    print("BUILDING MRMS DIAGNOSTIC IMAGE")
    print("=" * 72)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    ref, lats, lons = load("reflectivity")

    pflag, _, _ = load("precip_flag")
    rhohv, _, _ = load("rhohv")
    zdr, _, _ = load("zdr")

    bb_top, _, _ = load("bb_top")
    bb_bottom, _, _ = load("bb_bottom")

    rqi, _, _ = load("rqi")
    wetbulb, _, _ = load("wetbulb")
    freezing_level, _, _ = load("freezing_level")

    # --------------------------------------------------------
    # Clean data
    # --------------------------------------------------------

    ref = clean_field(ref, "reflectivity")
    pflag = clean_field(pflag, "precip_flag")
    rhohv = clean_field(rhohv, "rhohv")
    zdr = clean_field(zdr, "zdr")

    bb_top = clean_field(bb_top, "bb_top")
    bb_bottom = clean_field(bb_bottom, "bb_bottom")

    rqi = clean_field(rqi, "rqi")
    wetbulb = clean_field(wetbulb, "wetbulb")
    freezing_level = clean_field(
        freezing_level,
        "freezing_level",
    )

    # --------------------------------------------------------
    # Derived field
    # --------------------------------------------------------

    bb_depth = bb_top - bb_bottom

    # Don't display nonsense where either bright-band field is missing.
    invalid_bb = (
        ~np.isfinite(bb_top)
        | ~np.isfinite(bb_bottom)
        | (bb_depth < 0)
    )

    bb_depth[invalid_bb] = np.nan

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(19, 10),
        constrained_layout=True,
    )

    axes = axes.ravel()

    add_panel(
        axes[0],
        ref,
        lats,
        lons,
        "MRMS Reflectivity",
        "turbo",
        vmin=0,
        vmax=65,
        units="dBZ",
    )

    add_panel(
        axes[1],
        pflag,
        lats,
        lons,
        "MRMS PrecipFlag",
        "viridis",
        vmin=0,
        vmax=100,
        units="Flag",
    )

    add_panel(
        axes[2],
        rhohv,
        lats,
        lons,
        "MRMS RHOHV",
        "viridis",
        vmin=0.75,
        vmax=1.05,
        units="RHOHV",
    )

    add_panel(
        axes[3],
        zdr,
        lats,
        lons,
        "MRMS ZDR",
        "RdBu_r",
        vmin=-1,
        vmax=7,
        units="dB",
    )

    add_panel(
        axes[4],
        bb_depth,
        lats,
        lons,
        "Bright Band Depth",
        "magma",
        vmin=0,
        vmax=1500,
        units="m",
    )

    add_panel(
        axes[5],
        rqi,
        lats,
        lons,
        "Radar Quality Index",
        "viridis",
        vmin=0,
        vmax=1,
        units="RQI",
    )

    add_panel(
        axes[6],
        wetbulb,
        lats,
        lons,
        "Model Surface Wet-Bulb",
        "coolwarm",
        vmin=-15,
        vmax=15,
        units="°C",
    )

    add_panel(
        axes[7],
        freezing_level,
        lats,
        lons,
        "Model 0°C Height",
        "plasma",
        vmin=0,
        vmax=5000,
        units="m",
    )

    fig.suptitle(
        "MRMS Winter Radar — CONUS Diagnostic Fields",
        fontsize=20,
        fontweight="bold",
    )

    output = OUTPUT_DIR / "mrms_diagnostics.png"

    fig.savefig(
        output,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Wrote: {output}")
    print("")


if __name__ == "__main__":
    main()
