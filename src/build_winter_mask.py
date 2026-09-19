#!/usr/bin/env python3
"""
WinterRadar winter-phase diagnostic.

This is a diagnostic product only. It does NOT perform the final
precipitation-type classification.

The earlier implementation loaded the complete 3500 x 7000 RHOHV and
ZDR fields at all eight vertical levels at once. That produced a native
memory crash after the diagnostic image was written.

This version:
    * Reads one GRIB field at a time with pygrib.
    * Explicitly closes every GRIB handle.
    * Downsamples only the arrays needed for plotting.
    * Does not retain sixteen full-resolution vertical arrays.
    * Keeps the diagnostic focused on the fields used by the winter mask.

Outputs:
    outputs/winter_mask_diagnostics.png
    outputs/winter_mask_diagnostics.json
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

# Native-thread limits must be set before importing NumPy/Matplotlib.
for key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(key, "1")

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


FIELDS = {
    "Reflectivity":
        DATA_DIR /
        "MRMS_ReflectivityAtLowestAltitude.latest.grib2",

    "Reflectivity_0C":
        DATA_DIR /
        "MRMS_Reflectivity_0C.latest.grib2",

    "PrecipFlag":
        DATA_DIR /
        "MRMS_PrecipFlag.latest.grib2",

    "PrecipRate":
        DATA_DIR /
        "MRMS_PrecipRate.latest.grib2",

    "BrightBandTop":
        DATA_DIR /
        "MRMS_BrightBandTopHeight.latest.grib2",

    "BrightBandBottom":
        DATA_DIR /
        "MRMS_BrightBandBottomHeight.latest.grib2",

    "SurfaceTemp":
        DATA_DIR /
        "MRMS_Model_SurfaceTemp.latest.grib2",

    "WetBulbTemp":
        DATA_DIR /
        "MRMS_Model_WetBulbTemp.latest.grib2",

    "ZeroCHeight":
        DATA_DIR /
        "MRMS_Model_0degC_Height.latest.grib2",
}


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


INVALID = {
    "Reflectivity": (-999.0, -99.0, -3.0),
    "Reflectivity_0C": (-999.0, -99.0, -3.0),
    "PrecipFlag": (-999.0, -99.0, -3.0),
    "PrecipRate": (-999.0, -99.0, -3.0),
    "BrightBandTop": (-999.0, -99.0, -3.0),
    "BrightBandBottom": (-999.0, -99.0, -3.0),
    "SurfaceTemp": (-999.0, -99.0, -3.0),
    "WetBulbTemp": (-999.0, -99.0, -3.0),
    "ZeroCHeight": (-999.0, -99.0, -3.0),
}


def level_text(level: float) -> str:
    return f"{float(level):05.2f}"


def load_grib(path: Path) -> tuple[np.ndarray, str, str]:
    """
    Read one GRIB message and close the file before returning.
    """

    if not path.exists():
        raise FileNotFoundError(path)

    grbs = None
    grb = None

    try:
        grbs = pygrib.open(str(path))
        grb = grbs.message(1)

        data = np.array(
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

        return data, units, name

    finally:

        grb = None

        if grbs is not None:
            try:
                grbs.close()
            except Exception:
                pass

        gc.collect()


def clean(name: str, data: np.ndarray) -> np.ndarray:

    data = np.asarray(
        data,
        dtype=np.float32,
    )

    bad = ~np.isfinite(data)

    for value in INVALID.get(name, ()):

        bad |= np.isclose(
            data,
            value,
        )

    data[bad] = np.nan

    return data


def downsample(
    data: np.ndarray,
    target_rows: int = 700,
    target_cols: int = 1000,
) -> np.ndarray:

    rows, cols = data.shape

    row_step = max(
        1,
        int(np.ceil(rows / target_rows)),
    )

    col_step = max(
        1,
        int(np.ceil(cols / target_cols)),
    )

    return np.array(
        data[::row_step, ::col_step],
        dtype=np.float32,
        copy=True,
    )


def stats(data: np.ndarray) -> dict:

    finite = data[np.isfinite(data)]

    if finite.size == 0:

        return {
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
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "p05": float(np.percentile(finite, 5)),
        "p25": float(np.percentile(finite, 25)),
        "median": float(np.median(finite)),
        "p75": float(np.percentile(finite, 75)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
    }


def load_main_fields():

    fields = {}
    metadata = {}

    print()
    print("=" * 72)
    print("WINTERRADAR MRMS FIELD DIAGNOSTIC")
    print("=" * 72)

    for name, path in FIELDS.items():

        print()
        print("Loading", name)
        print(" ", path)

        try:

            data, units, grib_name = load_grib(
                path
            )

            data = clean(
                name,
                data
            )

            fields[name] = data

            field_stats = stats(
                data
            )

            metadata[name] = {
                "grib_name": grib_name,
                "units": units,
                "shape": list(data.shape),
                "statistics": field_stats,
            }

            print(
                f"  Shape: {data.shape}"
            )

            print(
                f"  Valid: {field_stats['count']:,}"
            )

            print(
                f"  Min: {field_stats['min']}"
            )

            print(
                f"  Max: {field_stats['max']}"
            )

        except Exception as exc:

            print(
                f"  ERROR: {exc}"
            )

    return fields, metadata


def vertical_signature():

    """
    Build only a low-memory vertical summary.

    We do NOT retain the full RHOHV/ZDR volume.

    The returned array is a reduced-resolution fraction of the eight
    levels meeting a conservative mixed-phase diagnostic signature.
    """

    print()
    print("=" * 72)
    print("VERTICAL DUAL-POL DIAGNOSTIC")
    print("=" * 72)

    fraction = None
    level_summaries = {}

    for level in LEVELS_KM:

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
            f"\nLoading RHOHV @ {level:.2f} km"
        )

        try:

            rho, rho_units, rho_name = load_grib(
                rho_path
            )

            rho = clean(
                "RHOHV",
                rho
            )

            print(
                f"  Valid: {np.count_nonzero(np.isfinite(rho)):,}"
            )

            print(
                f"  Shape: {rho.shape}"
            )

            rho_stats = stats(
                rho
            )

        except Exception as exc:

            print(
                f"  ERROR loading RHOHV: {exc}"
            )

            rho = None

            rho_stats = {
                "count": 0,
                "min": None,
                "p05": None,
                "p25": None,
                "median": None,
                "p75": None,
                "p95": None,
                "max": None,
            }

            rho_units = ""
            rho_name = ""

        print(
            f"Loading ZDR @ {level:.2f} km"
        )

        try:

            zdr, zdr_units, zdr_name = load_grib(
                zdr_path
            )

            zdr = clean(
                "ZDR",
                zdr
            )

            print(
                f"  Valid: {np.count_nonzero(np.isfinite(zdr)):,}"
            )

            print(
                f"  Shape: {zdr.shape}"
            )

            zdr_stats = stats(
                zdr
            )

        except Exception as exc:

            print(
                f"  ERROR loading ZDR: {exc}"
            )

            zdr = None

            zdr_stats = {
                "count": 0,
                "min": None,
                "p05": None,
                "p25": None,
                "median": None,
                "p75": None,
                "p95": None,
                "max": None,
            }

            zdr_units = ""
            zdr_name = ""

        if (
            rho is not None and
            zdr is not None
        ):

            signature = (
                np.isfinite(rho)
                &
                np.isfinite(zdr)
                &
                (rho < 0.98)
                &
                (zdr >= -1.0)
                &
                (zdr <= 4.0)
            )

            signature_display = downsample(
                signature.astype(np.float32)
            )

            if fraction is None:

                fraction = np.zeros_like(
                    signature_display,
                    dtype=np.float32,
                )

            fraction += signature_display

            level_summaries[
                f"{level:.2f}"
            ] = {
                "rhohv": rho_stats,
                "zdr": zdr_stats,
                "mixed_phase_signature_pixels":
                    int(np.count_nonzero(signature)),
            }

            del signature
            del signature_display

        else:

            level_summaries[
                f"{level:.2f}"
            ] = {
                "rhohv": rho_stats,
                "zdr": zdr_stats,
                "mixed_phase_signature_pixels": 0,
            }

        del rho
        del zdr

        gc.collect()

    if fraction is None:

        return (
            np.zeros(
                (350, 500),
                dtype=np.float32,
            ),
            level_summaries,
        )

    fraction /= float(
        len(LEVELS_KM)
    )

    return (
        fraction,
        level_summaries,
    )


def make_diagnostic_image(
    fields,
    vertical_fraction,
):

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 9),
        constrained_layout=True,
    )

    # ------------------------------------------------------------
    # Reflectivity
    # ------------------------------------------------------------

    ax = axes[0, 0]

    if "Reflectivity" in fields:

        data = downsample(
            fields["Reflectivity"]
        )

        finite = data[
            np.isfinite(data)
        ]

        if finite.size:

            im = ax.imshow(
                data,
                origin="upper",
                vmin=float(
                    np.percentile(
                        finite,
                        2,
                    )
                ),
                vmax=float(
                    np.percentile(
                        finite,
                        98,
                    )
                ),
                interpolation="nearest",
                aspect="auto",
            )

            fig.colorbar(
                im,
                ax=ax,
                shrink=0.78,
                label="dBZ",
            )

    ax.set_title(
        "MRMS Reflectivity"
    )

    # ------------------------------------------------------------
    # Precipitation flag
    # ------------------------------------------------------------

    ax = axes[0, 1]

    if "PrecipFlag" in fields:

        data = downsample(
            fields["PrecipFlag"]
        )

        im = ax.imshow(
            data,
            origin="upper",
            vmin=0,
            vmax=96,
            interpolation="nearest",
            aspect="auto",
        )

        fig.colorbar(
            im,
            ax=ax,
            shrink=0.78,
            label="Flag",
        )

    ax.set_title(
        "MRMS Precipitation Flag"
    )

    # ------------------------------------------------------------
    # Precipitation rate
    # ------------------------------------------------------------

    ax = axes[0, 2]

    if "PrecipRate" in fields:

        data = downsample(
            fields["PrecipRate"]
        )

        finite = data[
            np.isfinite(data)
        ]

        if finite.size:

            im = ax.imshow(
                data,
                origin="upper",
                vmin=0,
                vmax=max(
                    1.0,
                    float(
                        np.percentile(
                            finite,
                            99,
                        )
                    ),
                ),
                interpolation="nearest",
                aspect="auto",
            )

            fig.colorbar(
                im,
                ax=ax,
                shrink=0.78,
                label="Rate",
            )

    ax.set_title(
        "MRMS Precipitation Rate"
    )

    # ------------------------------------------------------------
    # Wet bulb
    # ------------------------------------------------------------

    ax = axes[1, 0]

    if "WetBulbTemp" in fields:

        data = downsample(
            fields["WetBulbTemp"]
        )

        finite = data[
            np.isfinite(data)
        ]

        if finite.size:

            im = ax.imshow(
                data,
                origin="upper",
                vmin=float(
                    np.percentile(
                        finite,
                        1,
                    )
                ),
                vmax=float(
                    np.percentile(
                        finite,
                        99,
                    )
                ),
                interpolation="nearest",
                aspect="auto",
            )

            fig.colorbar(
                im,
                ax=ax,
                shrink=0.78,
                label="°C",
            )

    ax.set_title(
        "MRMS Wet-Bulb Temperature"
    )

    # ------------------------------------------------------------
    # Zero C height
    # ------------------------------------------------------------

    ax = axes[1, 1]

    if "ZeroCHeight" in fields:

        data = downsample(
            fields["ZeroCHeight"]
        )

        finite = data[
            np.isfinite(data)
        ]

        if finite.size:

            im = ax.imshow(
                data,
                origin="upper",
                vmin=float(
                    np.percentile(
                        finite,
                        1,
                    )
                ),
                vmax=float(
                    np.percentile(
                        finite,
                        99,
                    )
                ),
                interpolation="nearest",
                aspect="auto",
            )

            fig.colorbar(
                im,
                ax=ax,
                shrink=0.78,
                label="m MSL",
            )

    ax.set_title(
        "MRMS 0°C Height"
    )

    # ------------------------------------------------------------
    # Vertical signature
    # ------------------------------------------------------------

    ax = axes[1, 2]

    im = ax.imshow(
        vertical_fraction,
        origin="upper",
        vmin=0,
        vmax=1,
        interpolation="nearest",
        aspect="auto",
    )

    fig.colorbar(
        im,
        ax=ax,
        shrink=0.78,
        label="Fraction of levels",
    )

    ax.set_title(
        "Vertical Mixed-Phase Signature"
    )

    fig.suptitle(
        "WinterRadar Precipitation-Phase Diagnostics",
        fontsize=17,
        fontweight="bold",
    )

    output = (
        OUTPUT_DIR /
        "winter_mask_diagnostics.png"
    )

    fig.savefig(
        output,
        dpi=130,
        facecolor="white",
        bbox_inches="tight",
    )

    plt.close(fig)

    del fig
    del axes

    gc.collect()

    print(
        f"Diagnostic image written: {output}"
    )


def write_json(
    fields,
    metadata,
    level_summaries,
):

    summary = {
        "description": (
            "WinterRadar precipitation-phase diagnostic. "
            "No final phase classification is performed."
        ),
        "mask_philosophy": {
            "rain": "transparent",
            "winter_precipitation": "masked",
            "mixed_precipitation": "masked",
            "no_precipitation": "transparent",
        },
        "fields": {},
        "vertical_dualpol": {
            "levels_km": LEVELS_KM,
            "levels": level_summaries,
        },
    }

    for name, data in fields.items():

        summary["fields"][name] = {
            "shape": list(data.shape),
            "statistics": stats(data),
            "units":
                metadata[name].get(
                    "units",
                    "",
                ),
        }

    output = (
        OUTPUT_DIR /
        "winter_mask_diagnostics.json"
    )

    output.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Diagnostic JSON written: {output}"
    )


def main():

    print()
    print("=" * 72)
    print("BUILDING WINTER PRECIPITATION DIAGNOSTIC")
    print("=" * 72)

    fields = {}
    metadata = {}
    level_summaries = {}

    try:

        fields, metadata = load_main_fields()

        vertical_fraction, level_summaries = (
            vertical_signature()
        )

        make_diagnostic_image(
            fields,
            vertical_fraction,
        )

        write_json(
            fields,
            metadata,
            level_summaries,
        )

        print()
        print("=" * 72)
        print("WINTERRADAR DIAGNOSTIC COMPLETE")
        print("=" * 72)

    finally:

        fields.clear()
        metadata.clear()
        level_summaries.clear()

        try:
            del vertical_fraction
        except Exception:
            pass

        gc.collect()


if __name__ == "__main__":
    main()
