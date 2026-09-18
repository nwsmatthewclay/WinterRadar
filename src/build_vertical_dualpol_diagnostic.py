from __future__ import annotations

from pathlib import Path

import cfgrib
import matplotlib.pyplot as plt
import numpy as np


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
DUALPOL_DIR = DATA_DIR / "dualpol"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

LEVELS_KM = [
    0.50,
    1.00,
    1.50,
    2.00,
    2.50,
    3.00,
    3.50,
    4.00,
]


# ------------------------------------------------------------
# File helpers
# ------------------------------------------------------------

def level_text(level: float) -> str:
    """
    Match the naming convention used by
    download_vertical_dualpol.py.

    0.50 -> 00.50
    1.00 -> 01.00
    4.00 -> 04.00
    """

    return f"{float(level):05.2f}"


def find_file(
    product: str,
    level: float,
) -> Path:

    text = level_text(level)

    path = (
        DUALPOL_DIR
        / f"{text}km"
        / (
            f"MRMS_{product}_"
            f"{text}.latest.grib2"
        )
    )

    if not path.exists():

        raise FileNotFoundError(
            "\n".join(
                [
                    "MRMS vertical dual-pol file "
                    "not found.",
                    "",
                    f"Product: {product}",
                    f"Level: {level:.2f} km",
                    f"Expected: {path}",
                ]
            )
        )

    return path


# ------------------------------------------------------------
# Load one level
# ------------------------------------------------------------

def load_level(
    product: str,
    level: float,
):

    path = find_file(
        product,
        level,
    )

    print("")
    print(
        f"Loading {product} "
        f"@ {level:.2f} km"
    )

    print(
        f"  {path}"
    )

    datasets = cfgrib.open_datasets(
        str(path),
        backend_kwargs={
            "indexpath": "",
        },
    )

    if not datasets:

        raise RuntimeError(
            f"No GRIB datasets found in {path}"
        )

    # Find the first dataset containing
    # an actual data variable.
    ds = None

    for candidate in datasets:

        if candidate.data_vars:

            ds = candidate
            break

    if ds is None:

        raise RuntimeError(
            f"No data variables found in {path}"
        )

    variable_name = list(
        ds.data_vars
    )[0]

    data = ds[variable_name].values

    # Convert masked arrays safely.
    if np.ma.isMaskedArray(data):

        data = data.filled(
            np.nan
        )

    data = np.asarray(
        data,
        dtype=np.float32,
    )

    # Latitude / longitude.
    if "latitude" in ds.coords:
        lats = ds.latitude.values

    elif "lat" in ds.coords:
        lats = ds.lat.values

    else:
        lats = None

    if "longitude" in ds.coords:
        lons = ds.longitude.values

    elif "lon" in ds.coords:
        lons = ds.lon.values

    else:
        lons = None

    print(
        f"  Variable: {variable_name}"
    )

    print(
        f"  Shape: {data.shape}"
    )

    valid = np.isfinite(data)

    if np.any(valid):

        print(
            f"  Valid: "
            f"{np.count_nonzero(valid):,}"
        )

        print(
            f"  Min: "
            f"{np.nanmin(data):.3f}"
        )

        print(
            f"  Max: "
            f"{np.nanmax(data):.3f}"
        )

        print(
            f"  Mean: "
            f"{np.nanmean(data):.3f}"
        )

    else:

        print(
            "  No valid data"
        )

    return (
        data,
        lats,
        lons,
    )


# ------------------------------------------------------------
# Build vertical diagnostic
# ------------------------------------------------------------

def main():

    print("")
    print("=" * 72)
    print(
        "BUILDING VERTICAL DUAL-POL DIAGNOSTIC"
    )
    print("=" * 72)

    rhohv_profiles = []
    zdr_profiles = []

    lats = None
    lons = None

    # --------------------------------------------------------
    # Load RHOHV
    # --------------------------------------------------------

    for level in LEVELS_KM:

        data, level_lats, level_lons = load_level(
            "MergedRhoHV",
            level,
        )

        rhohv_profiles.append(
            data
        )

        if lats is None:
            lats = level_lats
            lons = level_lons

    # --------------------------------------------------------
    # Load ZDR
    # --------------------------------------------------------

    for level in LEVELS_KM:

        data, _, _ = load_level(
            "MergedZdr",
            level,
        )

        zdr_profiles.append(
            data
        )

    # --------------------------------------------------------
    # Stack vertical levels
    # --------------------------------------------------------

    rhohv = np.stack(
        rhohv_profiles,
        axis=0,
    )

    zdr = np.stack(
        zdr_profiles,
        axis=0,
    )

    print("")
    print("=" * 72)
    print("VERTICAL ARRAYS")
    print("=" * 72)

    print(
        f"RHOHV shape: {rhohv.shape}"
    )

    print(
        f"ZDR shape:   {zdr.shape}"
    )

    # --------------------------------------------------------
    # Simple lowest-level diagnostic
    # --------------------------------------------------------

    rhohv_low = rhohv[0]
    zdr_low = zdr[0]

    # --------------------------------------------------------
    # Bright-band / mixed-phase indicators
    #
    # This is NOT the final precipitation classifier.
    # It is simply our first diagnostic visualization.
    # --------------------------------------------------------

    mixed_signal = (
        np.isfinite(rhohv_low)
        & np.isfinite(zdr_low)
        & (rhohv_low < 0.97)
        & (zdr_low > 0.5)
    )

    mixed_signal = mixed_signal.astype(
        np.uint8
    )

    print("")
    print(
        "Initial mixed-phase diagnostic:"
    )

    print(
        f"  Pixels flagged: "
        f"{np.count_nonzero(mixed_signal):,}"
    )

    # --------------------------------------------------------
    # Create diagnostic image
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6),
    )

    # RHOHV
    im0 = axes[0].imshow(
        rhohv_low,
        vmin=0.75,
        vmax=1.05,
        cmap="viridis",
    )

    axes[0].set_title(
        "MRMS RHOHV — 0.5 km"
    )

    axes[0].set_axis_off()

    fig.colorbar(
        im0,
        ax=axes[0],
        fraction=0.046,
        pad=0.04,
        label="RHOHV",
    )

    # ZDR
    im1 = axes[1].imshow(
        zdr_low,
        vmin=-2,
        vmax=6,
        cmap="turbo",
    )

    axes[1].set_title(
        "MRMS ZDR — 0.5 km"
    )

    axes[1].set_axis_off()

    fig.colorbar(
        im1,
        ax=axes[1],
        fraction=0.046,
        pad=0.04,
        label="dB",
    )

    # Mixed phase
    im2 = axes[2].imshow(
        mixed_signal,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[2].set_title(
        "Initial Mixed-Phase Signal"
    )

    axes[2].set_axis_off()

    fig.colorbar(
        im2,
        ax=axes[2],
        fraction=0.046,
        pad=0.04,
    )

    fig.suptitle(
        "MRMS Vertical Dual-Pol Diagnostic",
        fontsize=16,
    )

    fig.tight_layout()

    output = (
        OUTPUT_DIR
        / "vertical_dualpol_diagnostic.png"
    )

    fig.savefig(
        output,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("")
    print(
        f"Wrote {output}"
    )

    # --------------------------------------------------------
    # Save basic NPZ diagnostic data
    # --------------------------------------------------------

    npz_output = (
        OUTPUT_DIR
        / "vertical_dualpol.npz"
    )

    np.savez_compressed(
        npz_output,
        levels_km=np.asarray(
            LEVELS_KM,
            dtype=np.float32,
        ),
        rhohv=rhohv,
        zdr=zdr,
    )

    print(
        f"Wrote {npz_output}"
    )

    print("")
    print("=" * 72)
    print(
        "VERTICAL DUAL-POL DIAGNOSTIC COMPLETE"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
