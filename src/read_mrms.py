from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

import cfgrib
import numpy as np
import xarray as xr

from config import field_info


def product_from_path(path: Path) -> str:
    name = path.name
    prefix = "MRMS_"
    suffix = ".latest.grib2"
    if not name.startswith(prefix) or not name.endswith(suffix):
        raise ValueError(f"Cannot infer MRMS product from filename: {name}")
    return name[len(prefix) : -len(suffix)]


def read_grib(path: Path) -> xr.Dataset:
    if not path.exists():
        raise FileNotFoundError(path)

    datasets = cfgrib.open_datasets(
        str(path),
        backend_kwargs={"indexpath": ""},
    )
    if not datasets:
        raise RuntimeError(f"No GRIB messages found in {path}")
    return datasets[0]


def get_variable_name(ds: xr.Dataset) -> str:
    candidates = []
    for name, data in ds.data_vars.items():
        if name in {"latitude", "longitude"}:
            continue
        if getattr(data, "ndim", 0) >= 2:
            candidates.append(name)

    if not candidates:
        raise RuntimeError("No 2-D meteorological data variable found in GRIB dataset.")

    return candidates[0]


def clean_mrms_values(data: np.ndarray, product: str) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    if not arr.flags.writeable:
        arr = arr.copy()
    else:
        arr = arr.copy()

    info = field_info(product)
    for value in (*info["missing"], *info["no_coverage"]):
        arr[arr == value] = np.nan

    # Guard against the most common MRMS fill values even if a product's
    # metadata table changes independently of this code.
    arr[arr <= -900.0] = np.nan
    return arr


def get_values(
    path: Path,
    product: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one MRMS GRIB into float32 NumPy arrays and close the dataset."""
    if product is None:
        product = product_from_path(path)

    ds = read_grib(path)
    try:
        var_name = get_variable_name(ds)
        data = np.asarray(ds[var_name].values, dtype=np.float32).copy()
        lats = np.asarray(ds.latitude.values).copy()
        lons = np.asarray(ds.longitude.values).copy()
    finally:
        ds.close()

    if data.ndim != 2:
        raise RuntimeError(f"Expected a 2-D MRMS field, got shape {data.shape} from {path}")

    data = clean_mrms_values(data, product)
    return data, lats, lons



def _iso_from_grib_date_time(date_value, time_value) -> str | None:
    """Convert ecCodes/cfgrib date/time keys to an ISO-8601 UTC string."""
    if date_value is None or time_value is None:
        return None

    try:
        date_text = str(int(date_value)).zfill(8)
        time_text = str(int(time_value)).zfill(4)
        if len(time_text) > 4:
            # Some GRIB producers expose HHMMSS rather than HHMM.
            time_text = time_text[-6:].zfill(6)
        if len(time_text) == 4:
            hours = int(time_text[:2])
            minutes = int(time_text[2:4])
            seconds = 0
        else:
            hours = int(time_text[:2])
            minutes = int(time_text[2:4])
            seconds = int(time_text[4:6])
        dt = datetime.strptime(date_text, "%Y%m%d").replace(
            hour=hours,
            minute=minutes,
            second=seconds,
            tzinfo=timezone.utc,
        )
        return dt.isoformat()
    except (TypeError, ValueError, OverflowError):
        return None


def _datetime64_to_iso(value) -> str | None:
    """Convert a NumPy datetime64 value to an ISO UTC string."""
    try:
        arr = np.asarray(value)
        if arr.size == 0 or not np.issubdtype(arr.dtype, np.datetime64):
            return None
        scalar = arr.reshape(-1)[0]
        text = np.datetime_as_string(scalar.astype("datetime64[s]"), unit="s")
        if text == "NaT":
            return None
        return text + "+00:00"
    except (TypeError, ValueError, OverflowError):
        return None


def get_valid_time_utc(path: Path) -> str | None:
    """
    Read the MRMS valid time using cfgrib/ecCodes, not pygrib.

    The live core deliberately avoids pygrib because the current GitHub Actions
    environment produced a native `free(): invalid pointer` abort during
    pygrib shutdown even though the radar outputs had already been written.
    """
    ds = read_grib(path)
    try:
        # Prefer decoded xarray time coordinates because they can preserve
        # seconds when the source GRIB contains them.
        for coord_name in ("valid_time", "time"):
            if coord_name in ds.coords:
                iso = _datetime64_to_iso(ds[coord_name].values)
                if iso:
                    return iso

        # Fallback to GRIB validity keys exposed by cfgrib.
        containers = [ds.attrs]
        containers.extend(
            getattr(ds[var_name], "attrs", {}) for var_name in ds.data_vars
        )
        for attrs in containers:
            iso = _iso_from_grib_date_time(
                attrs.get("GRIB_validityDate"),
                attrs.get("GRIB_validityTime"),
            )
            if iso:
                return iso

            iso = _iso_from_grib_date_time(
                attrs.get("GRIB_dataDate"),
                attrs.get("GRIB_dataTime"),
            )
            if iso:
                return iso

        return None
    finally:
        ds.close()

def get_metadata(path: Path) -> dict:
    product = product_from_path(path)
    ds = read_grib(path)
    try:
        var_name = get_variable_name(ds)
        attrs = {}
        attrs.update(ds.attrs)
        attrs.update(ds[var_name].attrs)

        result = {
            "product": product,
            "variable": var_name,
            "expected_units": field_info(product)["units"],
        }

        interesting = (
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

        for key in interesting:
            if key in attrs:
                value = attrs[key]
                if isinstance(value, np.generic):
                    value = value.item()
                result[key] = value

        return result
    finally:
        ds.close()
