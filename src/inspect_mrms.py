from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import cfgrib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PRODUCTS, MRMS_BASE, field_info
from read_mrms import clean_mrms_values, get_variable_name


def safe_float(value):
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
    datasets = cfgrib.open_datasets(
        str(path),
        backend_kwargs={"indexpath": ""},
    )

    if not datasets:
        raise RuntimeError(f"No GRIB dataset found: {path}")

    return datasets[0]


def summarize(data: np.ndarray, product: str) -> dict:
    """Create statistics from cleaned MRMS data."""
    finite = np.isfinite(data)

    result = {
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "total_points": int(data.size),
        "valid_points": int(np.count_nonzero(finite)),
        "missing_fraction": float(
            1.0 - (np.count_nonzero(finite) / data.size)
        ),
        "units": field_info(product)["units"],
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

        if product == "PrecipFlag":
            unique, counts = np.unique(values, return_counts=True)

            result["flag_counts"] = [
                {
                    "flag": int(round(float(value))),
                    "count": int(count),
                }
                for value, count in zip(unique, counts)
            ]

    return result


def extract_metadata(ds, var_name: str, product: str) -> dict:
    """Extract useful GRIB metadata."""
    attrs = {}
    attrs.update(ds.attrs)
    attrs.update(ds[var_name].attrs)

    output = {
        "product": product,
        "variable": var_name,
        "expected_units": field_info(product)["units"],
    }

    keys = (
        "GRIB_shortName",
        "GRIB_name",
        "GRIB_units",
        "GRIB_cfName",
        "GRIB_cfVarName",
        "GRIB_paramId",
        "GRIB_typeOfLevel",
        "GRIB_gridType",
        "GRIB_Nx",
        "GRIB_Ny",
        "GRIB_dataDate",
        "GRIB_dataTime",
        "GRIB_validityDate",
        "GRIB_validityTime",
    )

    for key in keys:
        if key in attrs:
            value = attrs[key]

            if isinstance(value, np.generic):
                value = value.item()

            output[key] = value

    return output


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_time = datetime.now(timezone.utc).isoformat()

    print("")
    print("MRMS WINTER RADAR FIELD INSPECTOR")
    print("=" * 72)
    print(f"MRMS base: {MRMS_BASE}")
    print(f"Run time:  {run_time}")

    report = {
        "run_time_utc": run_time,
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

        item = {
            "name": name,
            "product": product,
            "path": str(path.relative_to(ROOT)),
        }

        if not path.exists():
            message = f"Missing file: {path}"
            print(f"ERROR: {message}")
            item["error"] = message
            report["fields"][name] = item
            continue

        try:
            file_size = path.stat().st_size
            ds = open_grib(path)
            var_name = get_variable_name(ds)

            raw = np.asarray(ds[var_name].values, dtype=np.float32)
            data = clean_mrms_values(raw, product)

            lats = np.asarray(ds.latitude.values)
            lons = np.asarray(ds.longitude.values)

            stats = summarize(data, product)
            metadata = extract_metadata(ds, var_name, product)

            item.update(
                {
                    "file_size_bytes": int(file_size),
                    "file_size": human_size(file_size),
                    "variable": var_name,
                    "metadata": metadata,
                    "summary": stats,
                    "latitude_shape": list(lats.shape),
                    "longitude_shape": list(lons.shape),
                }
            )

            print(f"File size:       {human_size(file_size)}")
            print(f"Variable:        {var_name}")
            print(f"Expected units:  {field_info(product)['units']}")
            print(f"Shape:           {data.shape}")
            print(f"Valid:           {stats['valid_points']:,}")
            print(f"Missing:         {stats['missing_fraction']:.2%}")
            print(f"Min:             {stats.get('min')}")
            print(f"Max:             {stats.get('max')}")
            print(f"Mean:            {stats.get('mean')}")

            if product == "PrecipFlag":
                print(
                    f"Flag counts:     "
                    f"{stats.get('flag_counts', [])}"
                )

            if reference_shape is None:
                reference_shape = data.shape
                reference_lat_shape = lats.shape
                reference_lon_shape = lons.shape

            same_shape = data.shape == reference_shape
            same_lat = lats.shape == reference_lat_shape
            same_lon = lons.shape == reference_lon_shape

            item["grid_status"] = (
                "OK"
                if same_shape and same_lat and same_lon
                else "MISMATCH"
            )

            print(f"Grid status:     {item['grid_status']}")

        except Exception as exc:
            print(f"ERROR: {exc}")
            item["error"] = str(exc)

        report["fields"][name] = item

    successful = [
        item
        for item in report["fields"].values()
        if "error" not in item
    ]

    report["grid"] = {
        "reference_shape": list(reference_shape)
        if reference_shape else None,
        "reference_latitude_shape": list(reference_lat_shape)
        if reference_lat_shape else None,
        "reference_longitude_shape": list(reference_lon_shape)
        if reference_lon_shape else None,
        "all_successful_fields_match": bool(successful)
        and all(
            item.get("grid_status") == "OK"
            for item in successful
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


if __name__ == "__main__":
    main()
