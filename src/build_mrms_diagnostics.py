from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PRODUCTS
from read_mrms import get_values


def load(name: str):
    """Load one configured, cleaned MRMS field."""
    product = PRODUCTS[name]
    path = DATA_DIR / f"MRMS_{product}.latest.grib2"

    if not path.exists():
        raise FileNotFoundError(f"Missing MRMS field: {path}")

    return get_values(path, product=product)


def downsample(arr: np.ndarray, max_dimension: int = 1400) -> np.ndarray:
    """Downsample only for plotting."""
    step = max(
        1,
        int(np.ceil(max(arr.shape) / max_dimension)),
    )

    return arr[::step, ::step]


def coordinate_extent(lats, lons):
    """Return geographic extent for 1-D MRMS coordinates."""
    lats = np.asarray(lats)
    lons = np.asarray(lons)

    return [
        float(np.nanmin(lons)),
        float(np.nanmax(lons)),
        float(np.nanmin(lats)),
        float(np.nanmax(lats)),
    ]


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
    """Draw one field."""
    plotted = downsample(data)
    masked = np.ma.masked_invalid(plotted)

    extent = coordinate_extent(lats, lons)

    origin = (
        "lower"
        if np.asarray(lats)[0] < np.asarray(lats)[-1]
        else "upper"
    )

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

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.15)

    cbar = plt.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    if units:
        cbar.set_label(units)

    return image


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("")
    print("BUILDING MRMS DIAGNOSTIC IMAGE")
    print("=" * 72)

    ref, lats, lons = load("reflectivity")
    pflag, _, _ = load("precip_flag")
    precip_rate, _, _ = load("precip_rate")
    rhohv, _, _ = load("rhohv")
    zdr, _, _ = load("zdr")
    bb_top, _, _ = load("bb_top")
    bb_bottom, _, _ = load("bb_bottom")
    rqi, _, _ = load("rqi")
    wetbulb, _, _ = load("wetbulb")
    freezing_level, _, _ = load("freezing_level")

    bb_depth = bb_top - bb_bottom

    invalid_bb = (
        ~np.isfinite(bb_top)
        | ~np.isfinite(bb_bottom)
        | (bb_depth < 0)
    )

    bb_depth[invalid_bb] = np.nan

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(18, 13),
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
        0,
        65,
        "dBZ",
    )

    add_panel(
        axes[1],
        pflag,
        lats,
        lons,
        "MRMS PrecipFlag",
        "viridis",
        0,
        96,
        "Flag",
    )

    add_panel(
        axes[2],
        precip_rate,
        lats,
        lons,
        "MRMS PrecipRate",
        "turbo",
        0,
        20,
        "mm/hr",
    )

    add_panel(
        axes[3],
        rhohv,
        lats,
        lons,
        "MRMS RHOHV",
        "viridis",
        0.75,
        1.05,
        "RHOHV",
    )

    add_panel(
        axes[4],
        zdr,
        lats,
        lons,
        "MRMS ZDR",
        "RdBu_r",
        -1,
        7,
        "dB",
    )

    add_panel(
        axes[5],
        bb_depth,
        lats,
        lons,
        "Bright Band Depth",
        "magma",
        0,
        1500,
        "m",
    )

    add_panel(
        axes[6],
        rqi,
        lats,
        lons,
        "Radar Quality Index",
        "viridis",
        0,
        1,
        "RQI",
    )

    add_panel(
        axes[7],
        wetbulb,
        lats,
        lons,
        "Model Surface Wet-Bulb",
        "coolwarm",
        -15,
        15,
        "°C",
    )

    add_panel(
        axes[8],
        freezing_level,
        lats,
        lons,
        "Model 0°C Height",
        "plasma",
        0,
        5000,
        "m",
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


if __name__ == "__main__":
    main()
