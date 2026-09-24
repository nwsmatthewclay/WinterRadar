from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pygrib

from config import field_info


def product_from_path(path: Path) -> str:
    name = path.name
    prefix = "MRMS_"
    suffix = ".latest.grib2"

    if not name.startswith(prefix) or not name.endswith(suffix):
        raise ValueError(f"Cannot infer MRMS product from filename: {name}")

    return name[len(prefix) : -len(suffix)]


def _read_one(path: Path):
    """
    Read one MRMS GRIB message with pygrib.

    Everything needed from the native GRIB handle is copied into ordinary
    NumPy arrays/Python values before the handle is closed. This avoids
    keeping native ecCodes objects alive after the file is released.
    """

    grbs = None
    grb = None

    try:
        grbs = pygrib.open(str(path))
        grb = grbs.message(1)

        # Copy the actual data out of the native GRIB object.
        data = np.asarray(
            grb.values,
            dtype=np.float32
        ).copy()

        # Copy coordinates out of the native GRIB object.
        lats, lons = grb.latlons()

        lats = np.asarray(
            lats,
            dtype=np.float64
        ).copy()

        lons = np.asarray(
            lons,
            dtype=np.float64
        ).copy()

        valid_time = None

        # Preferred MRMS validity time.
        try:
            date_value = int(grb["validityDate"])
            time_value = int(grb["validityTime"])

            dt = datetime.strptime(
                f"{date_value:08d}{time_value:04d}",
                "%Y%m%d%H%M",
            ).replace(tzinfo=timezone.utc)

            valid_time = dt.isoformat()

        except Exception:

            # Fallback to dataDate/dataTime + forecastTime.
            try:
                date_value = int(grb["dataDate"])
                time_value = int(grb["dataTime"])

                dt = datetime.strptime(
                    f"{date_value:08d}{time_value:04d}",
                    "%Y%m%d%H%M",
                ).replace(tzinfo=timezone.utc)

                try:
                    forecast = int(grb["forecastTime"])
                except Exception:
                    forecast = 0

                valid_time = (
                    dt + timedelta(hours=forecast)
                ).isoformat()

            except Exception:
                valid_time = None

        return data, lats, lons, valid_time

    finally:
        # Explicitly release the Python reference before closing the
        # underlying native GRIB handle.
        grb = None

        if grbs is not None:
            try:
                grbs.close()
            except Exception:
                pass

        grbs = None


def get_variable_name(ds) -> str:
    """
    Compatibility helper for callers that previously passed an
    xarray Dataset.
    """

    candidates = []

    for name in getattr(ds, "data_vars", {}):
        if name not in {"latitude", "longitude"}:
            candidates.append(name)

    if not candidates:
        raise RuntimeError(
            "No 2-D meteorological data variable found."
        )

    return candidates[0]


def clean_mrms_values(
    data: np.ndarray,
    product: str,
) -> np.ndarray:

    arr = np.asarray(
        data,
        dtype=np.float32
    ).copy()

    info = field_info(product)

    for value in (
        *info["missing"],
        *info["no_coverage"],
    ):
        arr[arr == value] = np.nan

    arr[arr <= -900.0] = np.nan

    return arr


def get_values(
    path: Path,
    product: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    if product is None:
        product = product_from_path(path)

    data, lats, lons, _ = _read_one(path)

    if data.ndim != 2:
        raise RuntimeError(
            f"Expected a 2-D MRMS field, "
            f"got shape {data.shape} from {path}"
        )

    return (
        clean_mrms_values(data, product),
        lats,
        lons,
    )


def get_values_with_time(
    path: Path,
    product: str | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    str | None,
]:

    if product is None:
        product = product_from_path(path)

    data, lats, lons, valid_time = _read_one(path)

    if data.ndim != 2:
        raise RuntimeError(
            f"Expected a 2-D MRMS field, "
            f"got shape {data.shape} from {path}"
        )

    return (
        clean_mrms_values(data, product),
        lats,
        lons,
        valid_time,
    )


def get_valid_time(path: Path) -> str | None:
    """
    Extract MRMS valid time through pygrib.

    No xarray/cfgrib import is required.
    """

    _, _, _, valid_time = _read_one(path)

    return valid_time


def get_metadata(path: Path) -> dict:

    product = product_from_path(path)

    grbs = None
    grb = None

    try:
        grbs = pygrib.open(str(path))
        grb = grbs.message(1)

        result = {
            "product": product,
            "variable": str(
                grb.get("shortName", "")
            ),
            "expected_units": field_info(product)["units"],
        }

        interesting = (
            "shortName",
            "name",
            "units",
            "cfName",
            "cfVarName",
            "paramId",
            "typeOfLevel",
            "gridType",
            "Nx",
            "Ny",
            "Ni",
            "Nj",
            "dataDate",
            "dataTime",
            "validityDate",
            "validityTime",
        )

        for key in interesting:

            try:
                value = grb[key]
            except Exception:
                continue

            if isinstance(value, np.generic):
                value = value.item()

            result[f"GRIB_{key}"] = value

        return result

    finally:

        grb = None

        if grbs is not None:
            try:
                grbs.close()
            except Exception:
                pass

        grbs = None


def read_grib(path: Path):
    """
    Compatibility reader.

    Core processing uses the detached NumPy arrays returned here rather
    than retaining an xarray/cfgrib Dataset or native GRIB handle.
    """

    data, lats, lons, valid_time = _read_one(path)

    return {
        "data": data,
        "latitude": lats,
        "longitude": lons,
        "valid_time": valid_time,
    }
