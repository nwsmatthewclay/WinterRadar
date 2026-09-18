from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import (
    DATA_DIR,
    VERTICAL_DUALPOL_LEVELS_KM,
)


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


def load_level(
    product: str,
    level_km: float,
):
    path = (
        DATA_DIR
        / "dualpol"
        / f"{level_km:.2f}km"
        / f"MRMS_{product}_{level_km:.2f}.latest.grib2"
    )

    import cfgrib

    datasets = cfgrib.open_datasets(
        str(path),
        backend_kwargs={"indexpath": ""},
    )

    ds = datasets[0]

    variables = [
        name
        for name in ds.data_vars
        if name not in {"latitude", "longitude"}
    ]

    if not variables:
        raise RuntimeError(
            f"No data variable found in {path}"
        )

    variable = variables[0]

    data = np.asarray(
        ds[variable].values,
        dtype=np.float32,
    )

    # MRMS missing values.
    data[
        (data <= -98.99)
    ] = np.nan

    return (
        data,
        ds.latitude.values,
        ds.longitude.values,
    )


def downsample(
    data,
    maximum_dimension=1200,
):
    step = max(
        1,
        int(
            np.ceil(
                max(data.shape)
                / maximum_dimension
            )
        ),
    )

    return data[::step, ::step]


def extent(
    lats,
    lons,
):
    return [
        float(np.nanmin(lons)),
        float(np.nanmax(lons)),
        float(np.nanmin(lats)),
        float(np.nanmax(lats)),
    ]


def plot_field(
    ax,
    data,
    lats,
    lons,
    title,
    cmap,
    vmin,
    vmax,
):
    plotted = downsample(data)

    origin = (
        "lower"
        if lats[0] < lats[-1]
        else "upper"
    )

    image = ax.imshow(
        np.ma.masked_invalid(plotted),
        extent=extent(lats, lons),
        origin=origin,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect="auto",
    )

    ax.set_title(
        title,
        fontsize=11,
        fontweight="bold",
    )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    ax.grid(
        True,
        alpha=0.15,
    )

    plt.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("")
    print("=" * 72)
    print("BUILDING VERTICAL DUAL-POL DIAGNOSTIC")
    print("=" * 72)

    fig, axes = plt.subplots(
        4,
        4,
        figsize=(18, 16),
        constrained_layout=True,
    )

    axes = axes.ravel()

    plot_index = 0

    # --------------------------------------------------------
    # RHOHV panels
    # --------------------------------------------------------

    for level_km in VERTICAL_DUALPOL_LEVELS_KM:

        data, lats, lons = load_level(
            "MergedRhoHV",
            level_km,
        )

        plot_field(
            axes[plot_index],
            data,
            lats,
            lons,
            f"RHOHV — {level_km:.2f} km",
            "viridis",
            0.75,
            1.05,
        )

        plot_index += 1

    # --------------------------------------------------------
    # ZDR panels
    # --------------------------------------------------------

    for level_km in VERTICAL_DUALPOL_LEVELS_KM:

        data, lats, lons = load_level(
            "MergedZdr",
            level_km,
        )

        plot_field(
            axes[plot_index],
            data,
            lats,
            lons,
            f"ZDR — {level_km:.2f} km",
            "RdBu_r",
            -2,
            7,
        )

        plot_index += 1

    fig.suptitle(
        "MRMS WinterRadar — Vertical Dual-Pol Structure",
        fontsize=20,
        fontweight="bold",
    )

    output = (
        OUTPUT_DIR
        / "mrms_vertical_dualpol.png"
    )

    fig.savefig(
        output,
        dpi=140,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Wrote: {output}")


if __name__ == "__main__":
    main()
