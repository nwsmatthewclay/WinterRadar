from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
import sys

import cfgrib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PRODUCTS, MRMS_BASE  # noqa: E402


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def safe_float(value):
    """Convert a value to a JSON-safe float."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(value):
        return None

    return value


def human_size(size_bytes: int) -> str:
    value = float(size_bytes)

    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024

    return f"{value:.1f} TB"


def open_grib(path: Path):
    """Open the first GRIB dataset."""
    datasets = cfgrib.open_datasets(
        str(path),
        backend_kwargs={"indexpath": ""},
    )

    if not datasets:
        raise RuntimeError(f"No GRIB dataset found: {path}")

    return datasets[0]


def get_primary_variable(ds):
    """Find the meteorological data variable."""
    variables = []

    for name in ds.data_vars:
        if name not in {"latitude", "longitude"}:
            variables.append(name)

    if not variables:
        raise RuntimeError("No meteorological data variable found.")

    return variables[0]


def mask_missing(product: str, data: np.ndarray) -> np.ndarray:
    """
    Convert known MRMS fill/sentinel values to NaN.

    We intentionally do not assume every product uses the same
    missing value.
    """
    data = np.asarray(data, dtype=np.float32).copy()

    if product in {
        "ReflectivityAtLowestAltitude",
        "MergedRhoHV",
        "MergedZdr",
        "Model_SurfaceTemp",
        "Model_WetBulbTemp",
    }:
        data[data <= -900] = np.nan

    elif product in {
        "BrightBandTopHeight",
        "BrightBandBottomHeight",
        "RadarQualityIndex",
        "Model_0degC_Height",
    }:
        data[data <= -2] = np.nan

    elif product == "PrecipFlag":
        data[data <= -2] = np.nan

    return data


def summarize(data: np.ndarray) -> dict:
    """Create useful statistics for one field."""
    finite = np.isfinite(data)

    result = {
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "total_points": int(data.size),
        "valid_points": int(np.count_nonzero(finite)),
        "missing_fraction": float(
            1.0 - (np.count_nonzero(finite) / data.size)
        ),
    }

    if np.any(finite):
        values = data[finite]

        result.update(
            {
                "min": safe_float(np.min(values)),
                "max": safe_float(np.max(values)),
                "mean": safe_float(np.mean(values)),
                "median": safe_float(np.median(values)),
                "p01": safe_float(np.percentile(values, 1)),
                "p05": safe_float(np.percentile(values, 5)),
                "p25": safe_float(np.percentile(values, 25)),
                "p50": safe_float(np.percentile(values, 50)),
                "p75": safe_float(np.percentile(values, 75)),
                "p95": safe_float(np.percentile(values, 95)),
                "p99": safe_float(np.percentile(values, 99)),
            }
        )

        # Helpful for categorical fields like PrecipFlag.
        if values.size <= 30_000_000:
            unique, counts = np.unique(values, return_counts=True)

            result["unique_values"] = [
                {
                    "value": safe_float(value),
                    "count": int(count),
                }
                for value, count in zip(unique[:200], counts[:200])
            ]

    return result


def extract_metadata(ds, var_name: str) -> dict:
    """Extract useful GRIB/xarray metadata."""
    attrs = dict(ds.attrs)

    if var_name in ds:
        attrs.update(ds[var_name].attrs)

    keep = [
        "GRIB_shortName",
        "GRIB_name",
        "GRIB_units",
        "GRIB_cfName",
        "GRIB_typeOfLevel",
        "GRIB_gridType",
        "GRIB_Nx",
        "GRIB_Ny",
        "GRIB_dataDate",
        "GRIB_dataTime",
        "GRIB_validityDate",
        "GRIB_validityTime",
    ]

    output = {}

    for key in keep:
        if key in attrs:
            value = attrs[key]

            if isinstance(value, np.generic):
                value = value.item()

            output[key] = value

    return output


# ------------------------------------------------------------
# Main inspector
# ------------------------------------------------------------

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("")
    print("MRMS WINTER RADAR FIELD INSPECTOR")
    print("=" * 72)
    print(f"MRMS base: {MRMS_BASE}")
    print(f"Run time:  {datetime.now(timezone.utc).isoformat()}")
    print("")

    report = {
        "run_time_utc": datetime.now(timezone.utc).isoformat(),
        "mrms_base": MRMS_BASE,
        "fields": {},
        "grid": {},
    }

    reference_shape = None
    reference_lat_shape = None
    reference_lon_shape = None

    for name, product in PRODUCTS.items():

        print("")
        print("-" * 72)
        print(f"{name.upper()} : {product}")
        print("-" * 72)

        path = DATA_DIR / f"MRMS_{product}.latest.grib2"

        field_report = {
            "name": name,
            "product": product,
            "path": str(path.relative_to(ROOT)),
        }

        if not path.exists():
            message = f"Missing file: {path}"
            print(f"ERROR: {message}")
            field_report["error"] = message
            report["fields"][name] = field_report
            continue

        try:
            size = path.stat().st_size

            ds = open_grib(path)
            var_name = get_primary_variable(ds)

            raw = np.asarray(ds[var_name].values, dtype=np.float32)
            data = mask_missing(product, raw)

            lats = np.asarray(ds.latitude.values)
            lons = np.asarray(ds.longitude.values)

            metadata = extract_metadata(ds, var_name)
            stats = summarize(data)

            field_report.update(
                {
                    "file_size_bytes": int(size),
                    "file_size": human_size(size),
                    "variable": var_name,
                    "metadata": metadata,
                    "summary": stats,
                    "latitude_shape": list(lats.shape),
                    "longitude_shape": list(lons.shape),
                }
            )

            print(f"File size:       {human_size(size)}")
            print(f"Variable:        {var_name}")
            print(f"Shape:           {data.shape}")
            print(f"Valid:           {stats['valid_points']:,}")
            print(f"Missing:         {stats['missing_fraction']:.2%}")
            print(f"Min:             {stats.get('min')}")
            print(f"Max:             {stats.get('max')}")
            print(f"Mean:            {stats.get('mean')}")

            if "GRIB_units" in metadata:
                print(f"Units:           {metadata['GRIB_units']}")

            if reference_shape is None:
                reference_shape = data.shape
                reference_lat_shape = lats.shape
                reference_lon_shape = lons.shape

            same_shape = data.shape == reference_shape
            same_lat = lats.shape == reference_lat_shape
            same_lon = lons.shape == reference_lon_shape

            status = (
                "OK"
                if same_shape and same_lat and same_lon
                else "MISMATCH"
            )

            field_report["grid_status"] = status

            print(f"Grid status:     {status}")

        except Exception as exc:
            print(f"ERROR: {exc}")
            field_report["error"] = str(exc)

        report["fields"][name] = field_report

    # --------------------------------------------------------
    # Grid summary
    # --------------------------------------------------------

    report["grid"] = {
        "reference_shape": list(reference_shape)
        if reference_shape else None,
        "reference_latitude_shape": list(reference_lat_shape)
        if reference_lat_shape else None,
        "reference_longitude_shape": list(reference_lon_shape)
        if reference_lon_shape else None,
        "all_fields_match": all(
            item.get("grid_status") == "OK"
            for item in report["fields"].values()
            if "error" not in item
        ),
    }

    output = OUTPUT_DIR / "mrms_diagnostics.json"

    output.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("")
    print("=" * 72)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 72)
    print(f"Wrote: {output}")
    print("")


if __name__ == "__main__":
    main()
