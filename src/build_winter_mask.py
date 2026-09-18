#!/usr/bin/env python3

"""
WinterRadar precipitation-phase diagnostic.

FIRST STAGE:
    Inspect the MRMS fields that will eventually be used to create
    a winter precipitation mask.

This version DOES NOT attempt to make a final precipitation-type
classification.

It:
    - Loads the available MRMS fields
    - Loads the vertical RHOHV/ZDR profiles
    - Prints useful statistics
    - Creates a diagnostic image
    - Writes a JSON summary

Important display philosophy:
    Rain will eventually be TRANSPARENT.
    Only winter/mixed precipitation will receive a mask.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import matplotlib.pyplot as plt
import cfgrib


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
DUALPOL_DIR = DATA_DIR / "dualpol"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# MRMS FIELDS
# ============================================================

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


# Vertical dual-pol levels downloaded by the workflow.
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


# ============================================================
# HELPERS
# ============================================================

def level_text(level):
    """
    Convert:
        0.50 -> 00.50
        1.00 -> 01.00
        4.00 -> 04.00
    """

    return f"{float(level):05.2f}"


def find_data_variable(ds):
    """Return the first actual data variable in an xarray dataset."""

    if not ds.data_vars:
        raise RuntimeError("No data variables found in GRIB dataset.")

    return next(iter(ds.data_vars))


def load_grib(path):
    """
    Load the first useful dataset/data variable from a GRIB2 file.

    Returns:
        array
        units
        attributes
    """

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    datasets = cfgrib.open_datasets(
        str(path),
        backend_kwargs={"indexpath": ""}
    )

    if not datasets:
        raise RuntimeError(f"No datasets found in {path}")

    selected = None

    for ds in datasets:
        if ds.data_vars:
            selected = ds
            break

    if selected is None:
        raise RuntimeError(f"No data variables found in {path}")

    variable = find_data_variable(selected)

    data = selected[variable].values

    # Handle masked arrays.
    if np.ma.isMaskedArray(data):
        data = np.ma.filled(data, np.nan)

    data = np.asarray(data, dtype=float)
    data = np.squeeze(data)

    units = selected[variable].attrs.get("units", "")

    attrs = dict(selected[variable].attrs)

    # Close datasets after loading.
    for ds in datasets:
        try:
            ds.close()
        except Exception:
            pass

    return data, units, attrs


def finite_values(data):
    """Return only finite values."""

    values = np.asarray(data, dtype=float).ravel()

    return values[np.isfinite(values)]


def statistics(data):
    """Calculate useful statistics."""

    values = finite_values(data)

    if values.size == 0:
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
        "count": int(values.size),
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def print_stats(name, data, units=""):
    """Print statistics in an easy-to-read format."""

    stats = statistics(data)

    unit_text = f" [{units}]" if units else ""

    print(f"\n{name}{unit_text}")

    if stats["count"] == 0:
        print("  NO FINITE DATA")
        return

    print(f"  Count : {stats['count']:,}")
    print(f"  Min   : {stats['min']:.3f}")
    print(f"  P05   : {stats['p05']:.3f}")
    print(f"  P25   : {stats['p25']:.3f}")
    print(f"  Median: {stats['median']:.3f}")
    print(f"  P75   : {stats['p75']:.3f}")
    print(f"  P95   : {stats['p95']:.3f}")
    print(f"  Max   : {stats['max']:.3f}")


def convert_temperature_to_celsius(data, units):
    """
    Convert temperature to Celsius when the source is Kelvin.

    If units aren't clearly Kelvin, leave the values unchanged.
    """

    if units and units.lower() in ("k", "kelvin"):
        return data - 273.15

    return data


# ============================================================
# MAIN MRMS DIAGNOSTIC
# ============================================================

def load_main_fields():

    fields = {}
    metadata = {}

    print("\n" + "=" * 70)
    print("WINTERRADAR MRMS FIELD DIAGNOSTIC")
    print("=" * 70)

    for name, path in FIELDS.items():

        print(f"\nLoading {name}")
        print(f"  {path}")

        try:
            data, units, attrs = load_grib(path)

            fields[name] = data
            metadata[name] = {
                "units": units,
                "shape": list(data.shape),
                "attributes": attrs,
            }

            print_stats(name, data, units)

        except Exception as exc:
            print(f"  ERROR: {exc}")

    return fields, metadata


# ============================================================
# VERTICAL DUAL-POL
# ============================================================

def load_vertical_dualpol():

    rhohv = {}
    zdr = {}

    print("\n" + "=" * 70)
    print("VERTICAL DUAL-POL PROFILE")
    print("=" * 70)

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
            / f"MRMS_MergedZDR_{text}.latest.grib2"
        )

        # -----------------------------
        # RHOHV
        # -----------------------------

        try:

            rho_data, rho_units, _ = load_grib(rho_path)

            rhohv[level] = rho_data

            print_stats(
                f"RHOHV {level:.2f} km",
                rho_data,
                rho_units
            )

        except Exception as exc:

            print(
                f"\nRHOHV {level:.2f} km ERROR: {exc}"
            )

        # -----------------------------
        # ZDR
        # -----------------------------

        try:

            zdr_data, zdr_units, _ = load_grib(zdr_path)

            zdr[level] = zdr_data

            print_stats(
                f"ZDR {level:.2f} km",
                zdr_data,
                zdr_units
            )

        except Exception as exc:

            print(
                f"\nZDR {level:.2f} km ERROR: {exc}"
            )

    return rhohv, zdr


# ============================================================
# TEMPERATURE DIAGNOSTIC
# ============================================================

def temperature_summary(fields, metadata):

    print("\n" + "=" * 70)
    print("TEMPERATURE DIAGNOSTIC")
    print("=" * 70)

    for name in ("SurfaceTemp", "WetBulbTemp"):

        if name not in fields:
            continue

        data = fields[name]

        units = metadata[name].get("units", "")

        temp_c = convert_temperature_to_celsius(
            data,
            units
        )

        print_stats(
            f"{name} (Celsius diagnostic)",
            temp_c,
            "degC"
        )


# ============================================================
# SIMPLE DIAGNOSTIC IMAGE
# ============================================================

def create_diagnostic_image(fields, rhohv, zdr):

    print("\nCreating diagnostic image...")

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10)
    )

    # --------------------------------------------------------
    # Reflectivity
    # --------------------------------------------------------

    ax = axes[0, 0]

    if "Reflectivity" in fields:

        im = ax.imshow(
            fields["Reflectivity"],
            origin="upper"
        )

        ax.set_title("MRMS Reflectivity")
        ax.set_xlabel("Grid X")
        ax.set_ylabel("Grid Y")

        fig.colorbar(
            im,
            ax=ax,
            shrink=0.85,
            label="dBZ"
        )

    else:

        ax.set_title("Reflectivity unavailable")

    # --------------------------------------------------------
    # Precipitation Rate
    # --------------------------------------------------------

    ax = axes[0, 1]

    if "PrecipRate" in fields:

        im = ax.imshow(
            fields["PrecipRate"],
            origin="upper"
        )

        ax.set_title("MRMS Precipitation Rate")
        ax.set_xlabel("Grid X")
        ax.set_ylabel("Grid Y")

        fig.colorbar(
            im,
            ax=ax,
            shrink=0.85,
            label="Precip Rate"
        )

    else:

        ax.set_title("Precipitation Rate unavailable")

    # --------------------------------------------------------
    # Wet Bulb Temperature
    # --------------------------------------------------------

    ax = axes[1, 0]

    if "WetBulbTemp" in fields:

        units = metadata_units(
            fields,
            "WetBulbTemp"
        )

        temp_c = convert_temperature_to_celsius(
            fields["WetBulbTemp"],
            units
        )

        im = ax.imshow(
            temp_c,
            origin="upper"
        )

        ax.set_title("MRMS Wet-Bulb Temperature")
        ax.set_xlabel("Grid X")
        ax.set_ylabel("Grid Y")

        fig.colorbar(
            im,
            ax=ax,
            shrink=0.85,
            label="°C"
        )

    else:

        ax.set_title("Wet-Bulb Temperature unavailable")

    # --------------------------------------------------------
    # Lowest-level RHOHV
    # --------------------------------------------------------

    ax = axes[1, 1]

    if rhohv:

        level = min(rhohv.keys())

        im = ax.imshow(
            rhohv[level],
            origin="upper",
            vmin=0.80,
            vmax=1.02
        )

        ax.set_title(
            f"RHOHV {level:.2f} km"
        )

        ax.set_xlabel("Grid X")
        ax.set_ylabel("Grid Y")

        fig.colorbar(
            im,
            ax=ax,
            shrink=0.85,
            label="RHOHV"
        )

    else:

        ax.set_title("RHOHV unavailable")

    fig.suptitle(
        "WinterRadar Precipitation-Phase Diagnostics",
        fontsize=18,
        fontweight="bold"
    )

    fig.tight_layout()

    output = OUTPUT_DIR / "winter_mask_diagnostics.png"

    fig.savefig(
        output,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Diagnostic image written: {output}")


def metadata_units(fields, name):
    """
    Placeholder used by the plotting routine.

    Temperature conversion is handled separately during loading,
    so this simply prevents the plot from failing if metadata is
    unavailable.
    """

    # MRMS model temperatures are expected to be Kelvin.
    if name in ("SurfaceTemp", "WetBulbTemp"):
        return "K"

    return ""


# ============================================================
# JSON OUTPUT
# ============================================================

def write_json(fields, metadata, rhohv, zdr):

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
            "rhohv": {},
            "zdr": {},
        },
    }

    for name, data in fields.items():

        summary["fields"][name] = {
            "shape": list(data.shape),
            "statistics": statistics(data),
            "units": metadata[name].get("units", ""),
        }

    for level, data in rhohv.items():

        summary["vertical_dualpol"]["rhohv"][
            f"{level:.2f}"
        ] = statistics(data)

    for level, data in zdr.items():

        summary["vertical_dualpol"]["zdr"][
            f"{level:.2f}"
        ] = statistics(data)

    output = OUTPUT_DIR / "winter_mask_diagnostics.json"

    with output.open("w") as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    print(f"Diagnostic JSON written: {output}")


# ============================================================
# MAIN
# ============================================================

def main():

    warnings.filterwarnings(
        "ignore",
        message=".*eccodes.*"
    )

    fields, metadata = load_main_fields()

    rhohv, zdr = load_vertical_dualpol()

    temperature_summary(
        fields,
        metadata
    )

    create_diagnostic_image(
        fields,
        rhohv,
        zdr
    )

    write_json(
        fields,
        metadata,
        rhohv,
        zdr
    )

    print("\n" + "=" * 70)
    print("WINTERRADAR DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
