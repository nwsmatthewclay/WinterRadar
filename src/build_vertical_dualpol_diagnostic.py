#!/usr/bin/env python3
"""
WinterRadar vertical dual-pol diagnostic.

Purpose:
    Build a diagnostic image from MRMS MergedRhoHV and MergedZdr
    at 0.50 through 4.00 km without retaining all 16 full-resolution
    arrays in memory.

Why this version exists:
    The previous implementation successfully wrote the PNG and NPZ,
    but then aborted with:

        double free or corruption (!prev)
        exit code 134

    The previous process retained:
        RHOHV shape = (8, 3500, 7000)
        ZDR   shape = (8, 3500, 7000)

    That is a very large native-memory workload for an Actions runner.
    This implementation reads each GRIB file individually with pygrib,
    copies the values into a NumPy array, creates only the reduced
    diagnostic data needed for plotting, closes the GRIB handle, and
    releases the array before moving to the next file.

Outputs:
    outputs/vertical_dualpol_diagnostic.png
    outputs/vertical_dualpol_diagnostic.json

Optional:
    outputs/vertical_dualpol.npz contains only reduced-resolution
    arrays suitable for research/inspection, not the original
    3500 x 7000 full-resolution fields.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

# Set native-library thread limits BEFORE importing NumPy/Matplotlib.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pygrib


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DUALPOL_DIR = DATA_DIR / "dualpol"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

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


def level_text(level: float) -> str:
    return f"{float(level):05.2f}"


def load_one_grib(path: Path):
    """
    Read one GRIB message, copy the values into NumPy memory, and
    explicitly close the pygrib handle before returning.
    """

    if not path.exists():
        raise FileNotFoundError(path)

    grbs = None
    grb = None

    try:
        grbs = pygrib.open(str(path))
        grb = grbs.message(1)

        values = np.array(
            grb.values,
            dtype=np.float32,
            copy=True,
        )

        units = getattr(
            grb,
            "units",
            "",
        ) or ""

        name = getattr(
            grb,
            "name",
            "",
        ) or ""

        return values, units, name

    finally:

        grb = None

        if grbs is not None:
            try:
                grbs.close()
            except Exception:
                pass

        gc.collect()


def clean(data: np.ndarray) -> np.ndarray:
    """
    Convert MRMS missing/sentinel values to NaN.
    """

    data = np.asarray(
        data,
        dtype=np.float32,
    )

    invalid = (
        ~np.isfinite(data)
        | np.isclose(data, -999.0)
        | np.isclose(data, -99.0)
        | np.isclose(data, -3.0)
    )

    data[invalid] = np.nan

    return data


def finite_stats(data: np.ndarray) -> dict:

    finite = data[np.isfinite(data)]

    if finite.size == 0:
        return {
            "count": 0,
            "min": None,
            "p05": None,
            "median": None,
            "p95": None,
            "max": None,
        }

    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "p05": float(np.percentile(finite, 5)),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
    }


def downsample(data: np.ndarray, target_rows=700, target_cols=1000):
    """
    Produce a display-only view.

    The full-resolution source array is immediately eligible for
    garbage collection after this function returns.
    """

    rows, cols = data.shape

    row_step = max(
        1,
        int(np.ceil(rows / target_rows)),
    )

    col_step = max(
        1,
        int(np.ceil(cols / target_cols)),
    )

    return data[
        ::row_step,
        ::col_step
    ]


def mixed_phase_signature(
    rhohv: np.ndarray,
    zdr: np.ndarray,
) -> np.ndarray:
    """
    Conservative diagnostic signature only.

    This is NOT the final precipitation-type classifier.

    It flags locations with:
        - valid RHOHV below approximately 0.98
        - valid ZDR in a modest positive range

    This preserves the purpose of the existing diagnostic: locating
    areas that deserve additional winter-phase investigation.
    """

    return (
        np.isfinite(rhohv)
        &
        np.isfinite(zdr)
        &
        (rhohv < 0.98)
        &
        (zdr >= -1.0)
        &
        (zdr <= 4.0)
    )


def process_level(level: float):
    text = level_text(level)

    rho_path = (
        DUALPOL_DIR
        / f"{text}km"
        / f"MRMS_MergedRhoHV_{text}.latest.grib2"
    )

    zdr_path = (
        DUALPOL_DIR
        / f"{text}km"
        / f"MRMS_MergedZdr_{text}.latest.grib2"
    )

    print(
        f"Loading MergedRhoHV @ {level:.2f} km"
    )

    rho, rho_units, rho_name = load_one_grib(
        rho_path
    )

    rho = clean(rho)

    print(
        f"  Shape: {rho.shape}"
    )

    rho_stats = finite_stats(rho)

    print(
        f"  Valid: {rho_stats['count']:,}"
    )

    print(
        f"  Min: {rho_stats['min']}"
    )

    print(
        f"  Max: {rho_stats['max']}"
    )

    rho_display = downsample(rho)

    print(
        f"  Loading MergedZdr @ {level:.2f} km"
    )

    zdr, zdr_units, zdr_name = load_one_grib(
        zdr_path
    )

    zdr = clean(zdr)

    print(
        f"  Shape: {zdr.shape}"
    )

    zdr_stats = finite_stats(zdr)

    print(
        f"  Valid: {zdr_stats['count']:,}"
    )

    print(
        f"  Min: {zdr_stats['min']}"
    )

    print(
        f"  Max: {zdr_stats['max']}"
    )

    zdr_display = downsample(zdr)

    mixed = mixed_phase_signature(
        rho,
        zdr,
    )

    mixed_count = int(
        np.count_nonzero(mixed)
    )

    # Downsample the diagnostic signature for plotting.
    mixed_display = downsample(
        mixed.astype(np.uint8)
    ).astype(bool)

    # Copy the display arrays. Then the original full-resolution arrays
    # can be released before moving to the next altitude.
    output = {
        "rho_display": np.array(
            rho_display,
            dtype=np.float32,
            copy=True,
        ),
        "zdr_display": np.array(
            zdr_display,
            dtype=np.float32,
            copy=True,
        ),
        "mixed_display": np.array(
            mixed_display,
            dtype=bool,
            copy=True,
        ),
        "rho_stats": rho_stats,
        "zdr_stats": zdr_stats,
        "mixed_count": mixed_count,
        "rho_units": rho_units,
        "zdr_units": zdr_units,
        "rho_name": rho_name,
        "zdr_name": zdr_name,
        "shape": list(rho.shape),
    }

    del rho
    del zdr
    del mixed
    del rho_display
    del zdr_display
    del mixed_display

    gc.collect()

    return output


def save_reduced_npz(results: dict):
    """
    Save reduced display-resolution arrays only.

    This replaces the previous full-resolution 16-array NPZ, which
    unnecessarily retained/serialized a huge amount of data.
    """

    arrays = {}

    for level, result in results.items():

        key = f"{level:.2f}".replace(".", "_")

        arrays[
            f"rhohv_{key}"
        ] = result["rho_display"]

        arrays[
            f"zdr_{key}"
        ] = result["zdr_display"]

        arrays[
            f"mixed_{key}"
        ] = result["mixed_display"]

    np.savez_compressed(
        OUTPUT_DIR / "vertical_dualpol.npz",
        **arrays,
    )

    del arrays

    gc.collect()


def make_diagnostic_image(results: dict):

    levels = list(
        results.keys()
    )

    # Use the lowest level for the two full-field diagnostic maps.
    low = levels[0]

    rho_low = results[
        low
    ]["rho_display"]

    zdr_low = results[
        low
    ]["zdr_display"]

    # Vertical fraction of levels showing the mixed signature.
    mixed_fraction = np.zeros_like(
        results[low]["mixed_display"],
        dtype=np.float32,
    )

    for level in levels:
        mixed_fraction += results[
            level
        ]["mixed_display"].astype(
            np.float32
        )

    mixed_fraction /= max(
        len(levels),
        1,
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10),
        constrained_layout=True,
    )

    # ------------------------------------------------------------
    # RHOHV
    # ------------------------------------------------------------

    ax = axes[0, 0]

    im = ax.imshow(
        rho_low,
        origin="upper",
        vmin=0.80,
        vmax=1.02,
        interpolation="nearest",
        aspect="auto",
    )

    ax.set_title(
        f"RHOHV {low:.2f} km"
    )

    ax.set_xlabel("Grid X")
    ax.set_ylabel("Grid Y")

    fig.colorbar(
        im,
        ax=ax,
        shrink=0.82,
        label="RHOHV",
    )

    # ------------------------------------------------------------
    # ZDR
    # ------------------------------------------------------------

    ax = axes[0, 1]

    im = ax.imshow(
        zdr_low,
        origin="upper",
        vmin=-2,
        vmax=6,
        interpolation="nearest",
        aspect="auto",
    )

    ax.set_title(
        f"ZDR {low:.2f} km"
    )

    ax.set_xlabel("Grid X")
    ax.set_ylabel("Grid Y")

    fig.colorbar(
        im,
        ax=ax,
        shrink=0.82,
        label="dB",
    )

    # ------------------------------------------------------------
    # Vertical mixed-signature fraction
    # ------------------------------------------------------------

    ax = axes[1, 0]

    im = ax.imshow(
        mixed_fraction,
        origin="upper",
        vmin=0,
        vmax=1,
        interpolation="nearest",
        aspect="auto",
    )

    ax.set_title(
        "Vertical Mixed-Phase Signature Fraction"
    )

    ax.set_xlabel("Grid X")
    ax.set_ylabel("Grid Y")

    fig.colorbar(
        im,
        ax=ax,
        shrink=0.82,
        label="Fraction of levels",
    )

    # ------------------------------------------------------------
    # Level summary
    # ------------------------------------------------------------

    ax = axes[1, 1]

    mixed_counts = [
        results[level][
            "mixed_count"
        ]
        for level in levels
    ]

    ax.plot(
        levels,
        mixed_counts,
        marker="o",
    )

    ax.set_title(
        "Mixed-Phase Signature Pixel Count by Height"
    )

    ax.set_xlabel(
        "Height (km)"
    )

    ax.set_ylabel(
        "Pixels"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    fig.suptitle(
        "WinterRadar Vertical Dual-Pol Diagnostic",
        fontsize=17,
        fontweight="bold",
    )

    output = (
        OUTPUT_DIR /
        "vertical_dualpol_diagnostic.png"
    )

    fig.savefig(
        output,
        dpi=130,
        facecolor="white",
        bbox_inches="tight",
    )

    plt.close(fig)

    del mixed_fraction
    del rho_low
    del zdr_low

    gc.collect()

    print(
        f"Wrote {output}"
    )


def write_json(results: dict):

    summary = {
        "description": (
            "Vertical MRMS RHOHV/ZDR diagnostic. "
            "Each GRIB field is processed individually and "
            "released before the next level is loaded. "
            "The mixed-phase signature is diagnostic only and "
            "is not a final precipitation-type classification."
        ),
        "levels_km": LEVELS_KM,
        "levels": {},
    }

    for level in LEVELS_KM:

        result = results[level]

        summary["levels"][
            f"{level:.2f}"
        ] = {
            "shape": result["shape"],
            "rhohv": result["rho_stats"],
            "zdr": result["zdr_stats"],
            "mixed_phase_signature_pixels":
                result["mixed_count"],
        }

    output = (
        OUTPUT_DIR /
        "vertical_dualpol_diagnostic.json"
    )

    output.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Wrote {output}"
    )


def main():

    print()
    print("=" * 72)
    print("BUILDING VERTICAL DUAL-POL DIAGNOSTIC")
    print("=" * 72)
    print()

    results = {}

    try:

        for level in LEVELS_KM:

            results[level] = process_level(
                level
            )

        print()
        print("=" * 72)
        print("VERTICAL DIAGNOSTIC SUMMARY")
        print("=" * 72)

        print(
            "Levels:",
            ", ".join(
                f"{level:.2f} km"
                for level in LEVELS_KM
            )
        )

        total_signature = sum(
            results[level]["mixed_count"]
            for level in LEVELS_KM
        )

        print(
            "Total level-signature pixels:",
            f"{total_signature:,}"
        )

        make_diagnostic_image(
            results
        )

        save_reduced_npz(
            results
        )

        write_json(
            results
        )

        print()
        print("=" * 72)
        print("VERTICAL DUAL-POL DIAGNOSTIC COMPLETE")
        print("=" * 72)

    finally:

        results.clear()
        gc.collect()


if __name__ == "__main__":
    main()
