from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import cfgrib
import numpy as np


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PRODUCTS, MRMS_BASE  # noqa: E402


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def human_size(num_bytes: int) -> str:
    """Return a readable file size."""
    value = float(num_bytes)

    for unit in ["B", "KB", "MB", "GB"]:
        if value < 1024.0:
            return f"{value:.1f} {unit}"
        value /= 1024.0

    return f"{value:.1f} TB"


def safe_number(value):
    """Convert numpy/scalar values into JSON-safe values."""
    if value is None:
        return None

    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)

    if math.isnan(value) or math.isinf(value):
        return None

    return value


def get_grib_dataset(path: Path):
    """Open the first GRIB dataset/group."""
    datasets = cfgrib.open_datasets(
        str(path),
        backend_kwargs={"indexpath": ""},
    )

    if not datasets:
        raise RuntimeError(f"No GRIB datasets found in {path}")

    return datasets[0]


def get_data_variable(ds):
    """Return the primary meteorological data variable."""
    candidates = [
        name
        for name in ds.data_vars
        if name not in {"latitude", "longitude"}
    ]

    if not candidates:
        raise RuntimeError("Could not identify meteorological variable")

    return candidates[0]


def summarize_array(data: np.ndarray) -> dict:
    """Summarize a meteorological array."""
    data = np.asarray(data)

    finite = np.isfinite(data)

    result = {
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "finite_points": int(np.count_nonzero(finite)),
        "total_points": int(data.size),
        "missing_fraction": float(1.0 - (np.count_nonzero(finite) / data.size)),
    }

    if np.any(finite):
        valid = data[finite]

        result["min"] = safe_number(np.min(valid))
        result["max"] = safe_number(np.max(valid))
        result["mean"] = safe_number(np.mean(valid))
        result["median"] = safe_number(np.median(valid))

        # Useful percentiles for reflectivity and dual-pol fields.
        result["p01"] = safe_number(np.percentile(valid, 1))
        result["p05"] = safe_number(np.percentile(valid, 5))
        result["p25"] = safe_number(np.percentile(valid, 25))
        result["p50"] = safe_number(np.percentile(valid, 50))
        result["p75"] = safe_number(np.percentile(valid, 75))
        result["p95"] = safe_number(np.percentile(valid, 95))
        result["p99"] = safe_number(np.percentile(valid, 99))

        # Unique values are useful for categorical fields such as PrecipFlag.
        if valid.size < 5_000_000:
            unique_values, counts = np.unique(valid, return_counts=True)

            pairs = []

            for value, count in zip(unique_values[:100], counts[:100]):
                pairs.append({
                    "value": safe_number(value),
                    "count": int(count),
                })

            result["unique_values_sample"] = pairs

    return result


def extract_grib_metadata(ds) -> dict:
    """Extract useful GRIB metadata."""
    attrs = ds.attrs.copy()

    interesting_keys = [
        "GRIB_dataDate",
        "GRIB_dataTime",
        "GRIB_validityDate",
        "GRIB_validityTime",
        "GRIB_shortName",
        "GRIB_name",
        "GRIB_units",
        "GRIB_typeOfLevel",
        "GRIB_gridType",
        "GRIB_Nx",
        "GRIB_Ny",
        "GRIB_La1",
        "GRIB_Lo1",
        "GRIB_La2",
        "GRIB_Lo2",
        "GRIB_dxInMetres",
        "GRIB_dyInMetres",
        "GRIB_missingValue",
        "GRIB_cfName",
    ]

    metadata = {}

    for key in interesting_keys:
        if key in attrs:
            value = attrs[key]

            if isinstance(value, np.generic):
                value = value.item()

            metadata[key] = value

    return metadata


def get_sample_coordinates(ds) -> dict:
    """Return corner/center samples without loading unnecessary data."""
    lat = np.asarray(ds["latitude"].values)
    lon = np.asarray(ds["longitude"].values)

    samples = {}

    if lat.ndim == 1:
        lat_points = {
            "first": safe_number(lat[0]),
            "middle": safe_number(lat[len(lat) // 2]),
            "last": safe_number(lat[-1]),
        }

        lon_points = {
            "first": safe_number(lon[0]),
            "middle": safe_number(lon[len(lon) // 2]),
            "last": safe_number(lon[-1]),
        }

        samples["latitude"] = lat_points
        samples["longitude"] = lon_points

    else:
        y_mid = lat.shape[0] // 2
        x_mid = lat.shape[1] // 2

        samples["corners"] = {
            "upper_left": {
                "lat": safe_number(lat[0, 0]),
                "lon": safe_number(lon[0, 0]),
            },
            "upper_right": {
                "lat": safe_number(lat[0, -1]),
                "lon": safe_number(lon[0, -1]),
            },
            "lower_left": {
                "lat": safe_number(lat[-1, 0]),
                "lon": safe_number(lon[-1, 0]),
            },
            "lower_right": {
                "lat": safe_number(lat[-1, -1]),
                "lon": safe_number(lon[-1, -1]),
            },
            "center": {
                "lat": safe_number(lat[y_mid, x_mid]),
                "lon": safe_number(lon[y_mid, x_mid]),
            },
        }

    samples["latitude_shape"] = list(lat.shape)
    samples["longitude_shape"] = list(lon.shape)

    return samples


def inspect_field(name: str, product: str) -> dict:
    """Inspect one MRMS field."""
    path = DATA_DIR / f"MRMS_{product}.latest.grib2"

    print("")
    print("=" * 70)
    print(f"{name.upper()} : {product}")
    print("=" * 70)

    if not path.exists():
        raise FileNotFoundError(f"Missing GRIB file: {path}")

    file_size = path.stat().st_size

    print(f"File:       {path}")
    print(f"File size:  {human_size(file_size)}")

    ds = get_grib_dataset(path)
    var_name = get_data_variable(ds)

    print(f"Variable:   {var_name}")

    data = np.asarray(ds[var_name].values)

    print(f"Shape:      {data.shape}")
    print(f"Data type:  {data.dtype}")

    summary = summarize_array(data)

    print(f"Min:        {summary.get('min')}")
    print(f"Max:        {summary.get('max')}")
    print(f"Mean:       {summary.get('mean')}")
    print(f"Missing:    {summary['missing_fraction']:.2%}")

    metadata = extract_grib_metadata(ds)
    coordinates = get_sample_coordinates(ds)

    if "GRIB_dataDate" in metadata:
        print(
            f"Data time:  "
            f"{metadata.get('GRIB_dataDate')} "
            f"{metadata.get('GRIB_dataTime')}"
        )

    if "GRIB_validityDate" in metadata:
        print(
            f"Valid time: "
            f"{metadata.get('GRIB_validityDate')} "
            f"{metadata.get('GRIB_validityTime')}"
        )

    lat_shape = tuple(coordinates["latitude_shape"])
    lon_shape = tuple(coordinates["longitude_shape"])

    print(f"Lat shape:  {lat_shape}")
    print(f"Lon shape:  {lon_shape}")

    result = {
        "name": name,
        "product": product,
        "file": str(path.relative_to(ROOT)),
        "file_size_bytes": int(file_size),
        "file_size_human": human_size(file_size),
        "variable": var_name,
        "summary": summary,
        "metadata": metadata,
        "coordinates": coordinates,
    }

    return result


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("")
    print("MRMS WINTER RADAR FIELD INSPECTOR")
    print("=" * 70)
    print(f"MRMS base: {MRMS_BASE}")
    print(f"Run time:  {datetime.now(timezone.utc).isoformat()}")
    print("")

    results = {
        "run_time_utc": datetime.now(timezone.utc).isoformat(),
        "mrms_base": MRMS_BASE,
        "fields": {},
        "grid_comparison": {},
    }

    # These fields need to be downloaded by download_mrms.py first.
    for name, product in PRODUCTS.items():
        try:
            results["fields"][name] = inspect_field(name, product)
        except Exception as exc:
            print("")
            print(f"ERROR inspecting {name}: {exc}")
            results["fields"][name] = {
                "name": name,
                "product": product,
                "error": str(exc),
            }

    # --------------------------------------------------------
    # Compare grid geometry
    # --------------------------------------------------------

    print("")
    print("=" * 70)
    print("GRID CONSISTENCY CHECK")
    print("=" * 70)

    reference_shape = None
    reference_lat_shape = None
    reference_lon_shape = None

    grid_results = {}

    for name, result in results["fields"].items():

        if "error" in result:
            grid_results[name] = {
                "same_data_shape": False,
                "same_lat_shape": False,
                "same_lon_shape": False,
                "status": "ERROR",
            }
            continue

        data_shape = tuple(result["summary"]["shape"])
        lat_shape = tuple(result["coordinates"]["latitude_shape"])
        lon_shape = tuple(result["coordinates"]["longitude_shape"])

        if reference_shape is None:
            reference_shape = data_shape
            reference_lat_shape = lat_shape
            reference_lon_shape = lon_shape

        same_data = data_shape == reference_shape
        same_lat = lat_shape == reference_lat_shape
        same_lon = lon_shape == reference_lon_shape

        status = "OK" if (same_data and same_lat and same_lon) else "MISMATCH"

        grid_results[name] = {
            "data_shape": list(data_shape),
            "same_data_shape_as_reference": same_data,
            "same_lat_shape_as_reference": same_lat,
            "same_lon_shape_as_reference": same_lon,
            "status": status,
        }

        print(
            f"{name:18s} "
            f"data={str(data_shape):18s} "
            f"lat={str(lat_shape):18s} "
            f"lon={str(lon_shape):18s} "
            f"{status}"
        )

    results["grid_comparison"] = {
        "reference_data_shape": list(reference_shape)
        if reference_shape else None,
        "reference_lat_shape": list(reference_lat_shape)
        if reference_lat_shape else None,
        "reference_lon_shape": list(reference_lon_shape)
        if reference_lon_shape else None,
        "fields": grid_results,
    }

    # --------------------------------------------------------
    # Write JSON report
    # --------------------------------------------------------

    output_path = OUTPUT_DIR / "mrms_diagnostics.json"

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, allow_nan=False)

    print("")
    print("=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)
    print(f"Wrote: {output_path}")
    print("")


if __name__ == "__main__":
    main()
