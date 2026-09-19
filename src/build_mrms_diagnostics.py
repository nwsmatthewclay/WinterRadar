#!/usr/bin/env python3
"""
WinterRadar MRMS diagnostic image builder.

This version intentionally avoids cfgrib/xarray for the diagnostic image.
It uses pygrib one file at a time, explicitly closes every GRIB handle,
limits the amount of data retained in memory, and downsamples only for
display.  This is designed to avoid native-library shutdown crashes
observed after matplotlib finished writing the PNG.

Outputs:
    outputs/mrms_diagnostics.png
    outputs/mrms_diagnostics.json
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import matplotlib

# Force a non-interactive backend before importing pyplot.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pygrib


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


FIELDS = {
    "Reflectivity": DATA_DIR / "MRMS_ReflectivityAtLowestAltitude.latest.grib2",
    "Reflectivity_0C": DATA_DIR / "MRMS_Reflectivity_0C.latest.grib2",
    "PrecipFlag": DATA_DIR / "MRMS_PrecipFlag.latest.grib2",
    "PrecipRate": DATA_DIR / "MRMS_PrecipRate.latest.grib2",
    "BrightBandTop": DATA_DIR / "MRMS_BrightBandTopHeight.latest.grib2",
    "BrightBandBottom": DATA_DIR / "MRMS_BrightBandBottomHeight.latest.grib2",
    "SurfaceTemp": DATA_DIR / "MRMS_Model_SurfaceTemp.latest.grib2",
    "WetBulbTemp": DATA_DIR / "MRMS_Model_WetBulbTemp.latest.grib2",
    "ZeroCHeight": DATA_DIR / "MRMS_Model_0degC_Height.latest.grib2",
}


INVALID_VALUES = {
    "Reflectivity": (-999.0, -99.0),
    "Reflectivity_0C": (-999.0, -99.0),
    "PrecipFlag": (-999.0, -99.0, -3.0),
    "PrecipRate": (-999.0, -99.0, -3.0),
    "BrightBandTop": (-999.0, -99.0, -3.0),
    "BrightBandBottom": (-999.0, -99.0, -3.0),
    "SurfaceTemp": (-999.0, -99.0, -3.0),
    "WetBulbTemp": (-999.0, -99.0, -3.0),
    "ZeroCHeight": (-999.0, -99.0, -3.0),
}


def clean_array(name: str, data: np.ndarray) -> np.ndarray:
    """
    Return float32 data with known MRMS sentinel values converted to NaN.
    """
    arr = np.asarray(data, dtype=np.float32)

    bad = ~np.isfinite(arr)

    for value in INVALID_VALUES.get(name, ()):
        bad |= np.isclose(arr, value)

    arr[bad] = np.nan

    return arr


def load_grib_one_message(path: Path):
    """
    Read the first GRIB message and close the file explicitly.

    MRMS 2-D products used here contain a single field/message for the
    diagnostic purpose.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    grbs = None
    grb = None

    try:
        grbs = pygrib.open(str(path))
        grb = grbs.message(1)

        values = grb.values
        units = getattr(grb, "units", "") or ""
        name = getattr(grb, "name", "") or ""

        # Force a private NumPy copy so the GRIB handle can close safely.
        data = np.array(
            values,
            dtype=np.float32,
            copy=True,
        )

        return data, units, name

    finally:
        # Explicitly drop message before closing the GRIB file.
        grb = None

        if grbs is not None:
            try:
                grbs.close()
            except Exception:
                pass

        gc.collect()


def statistics(name: str, data: np.ndarray, units: str) -> dict:
    arr = clean_array(name, data)

    finite = arr[np.isfinite(arr)]

    if finite.size == 0:
        return {
            "units": units,
            "shape": list(arr.shape),
            "count": 0,
            "min": None,
            "p05": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p95": None,
            "max": None,
        }

    return {
        "units": units,
        "shape": list(arr.shape),
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "p05": float(np.percentile(finite, 5)),
        "p25": float(np.percentile(finite, 25)),
        "median": float(np.median(finite)),
        "p75": float(np.percentile(finite, 75)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
    }


def display_stride(shape, target_rows=700, target_cols=1000):
    rows, cols = shape

    row_step = max(1, int(np.ceil(rows / target_rows)))
    col_step = max(1, int(np.ceil(cols / target_cols)))

    return row_step, col_step


def display_array(data: np.ndarray) -> np.ndarray:
    """
    Downsample only for plotting. The full-resolution array is never
    duplicated for the displayed diagnostic.
    """
    row_step, col_step = display_stride(data.shape)

    if row_step == 1 and col_step == 1:
        return data

    return data[::row_step, ::col_step]


def finite_percentile(data, pct, default):
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return default
    return float(np.percentile(finite, pct))


def make_diagnostic_image(fields: dict[str, np.ndarray]) -> None:

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10),
        constrained_layout=True,
    )

    plots = [
        (
            axes[0, 0],
            "Reflectivity",
            "MRMS Reflectivity",
            "dBZ",
        ),
        (
            axes[0, 1],
            "PrecipRate",
            "MRMS Precipitation Rate",
            "Precipitation Rate",
        ),
        (
            axes[1, 0],
            "WetBulbTemp",
            "MRMS Wet-Bulb Temperature",
            "°C",
        ),
        (
            axes[1, 1],
            "ZeroCHeight",
            "MRMS 0°C Height",
            "m MSL",
        ),
    ]

    for ax, key, title, label in plots:

        if key not in fields:
            ax.set_title(title + " — unavailable")
            ax.axis("off")
            continue

        data = clean_array(
            key,
            fields[key],
        )

        shown = display_array(data)

        finite = shown[np.isfinite(shown)]

        if finite.size == 0:
            ax.set_title(title + " — no valid data")
            ax.axis("off")
            continue

        # Keep extreme invalid/outlier values from wrecking the display.
        if key in ("Reflectivity", "Reflectivity_0C"):
            vmin = finite_percentile(shown, 2, -20)
            vmax = finite_percentile(shown, 98, 65)
        elif key in ("PrecipRate",):
            vmin = 0.0
            vmax = finite_percentile(shown, 99, 10)
            vmax = max(vmax, 1.0)
        elif key in ("WetBulbTemp", "SurfaceTemp"):
            vmin = finite_percentile(shown, 1, -20)
            vmax = finite_percentile(shown, 99, 35)
        else:
            vmin = finite_percentile(shown, 1, 0)
            vmax = finite_percentile(shown, 99, 6000)

        im = ax.imshow(
            shown,
            origin="upper",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            aspect="auto",
        )

        ax.set_title(title)
        ax.set_xlabel("Grid X")
        ax.set_ylabel("Grid Y")

        fig.colorbar(
            im,
            ax=ax,
            shrink=0.82,
            label=label,
        )

    fig.savefig(
        OUTPUT_DIR / "mrms_diagnostics.png",
        dpi=130,
        facecolor="white",
        bbox_inches="tight",
    )

    plt.close(fig)

    # Help release matplotlib / NumPy objects before interpreter shutdown.
    gc.collect()


def main():

    # Keep common numerical libraries single-threaded in the diagnostic
    # process. This avoids unnecessary native-thread pressure on the runner.
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(key, "1")

    print("=" * 72)
    print("BUILDING MRMS DIAGNOSTIC IMAGE")
    print("=" * 72)

    fields: dict[str, np.ndarray] = {}
    metadata: dict[str, dict] = {}

    # Load each field one at a time. We retain the arrays because the
    # diagnostic uses four of them, but each GRIB file is explicitly
    # closed immediately after reading.
    for key, path in FIELDS.items():

        print(f"Loading {key}")
        print(f"  {path}")

        try:
            data, units, grib_name = load_grib_one_message(path)
            cleaned = clean_array(key, data)

            fields[key] = cleaned

            metadata[key] = {
                "grib_name": grib_name,
                "units": units,
                "statistics": statistics(
                    key,
                    cleaned,
                    units,
                ),
            }

            stats = metadata[key]["statistics"]

            print(
                f"  Shape: {stats['shape']}"
            )

            print(
                f"  Valid: {stats['count']:,}"
            )

        except Exception as exc:
            print(
                f"  ERROR: {exc}"
            )

    # Only four arrays are needed by the image after loading.
    make_diagnostic_image(fields)

    summary = {
        "description": (
            "MRMS core-field diagnostic. "
            "The diagnostic image uses a downsampled display copy; "
            "the source MRMS fields remain at full resolution while "
            "each GRIB file is explicitly closed after reading."
        ),
        "fields": metadata,
    }

    json_path = (
        OUTPUT_DIR /
        "mrms_diagnostics.json"
    )

    json_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Release references before interpreter shutdown.
    fields.clear()
    metadata.clear()
    gc.collect()

    print()
    print(
        "Wrote:",
        OUTPUT_DIR / "mrms_diagnostics.png",
    )

    print(
        "Wrote:",
        OUTPUT_DIR / "mrms_diagnostics.json",
    )

    print()
    print("=" * 72)
    print("MRMS DIAGNOSTIC COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
