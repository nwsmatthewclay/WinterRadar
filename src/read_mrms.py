from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

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
    arr = np.asarray(data, dtype=np.float32).copy()
    info = field_info(product)
    for value in (*info["missing"], *info["no_coverage"]):
        arr[arr == value] = np.nan
    arr[arr <= -900.0] = np.nan
    return arr


def get_values(
    path: Path,
    product: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    return clean_mrms_values(data, product), lats, lons


def get_values_with_time(
    path: Path,
    product: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str | None]:
    """Read one MRMS GRIB once and return data, coordinates, and valid time."""
    if product is None:
        product = product_from_path(path)

    ds = read_grib(path)
    try:
        var_name = get_variable_name(ds)
        data = np.asarray(ds[var_name].values, dtype=np.float32).copy()
        lats = np.asarray(ds.latitude.values).copy()
        lons = np.asarray(ds.longitude.values).copy()
        valid_time = None
        for key in ("valid_time", "time"):
            if key in ds.coords:
                valid_time = _as_utc_iso(ds.coords[key].values)
                if valid_time:
                    break
        if valid_time is None:
            attrs = {}
            attrs.update(ds.attrs)
            attrs.update(ds[var_name].attrs)
            date_value = attrs.get("GRIB_validityDate", attrs.get("GRIB_dataDate"))
            time_value = attrs.get("GRIB_validityTime", attrs.get("GRIB_dataTime"))
            if date_value is not None and time_value is not None:
                date_text = str(int(date_value)).zfill(8)
                time_text = str(int(time_value)).zfill(6)
                dt = datetime.strptime(date_text + time_text, "%Y%m%d%H%M%S")
                valid_time = dt.replace(tzinfo=timezone.utc).isoformat()
    finally:
        ds.close()

    if data.ndim != 2:
        raise RuntimeError(f"Expected a 2-D MRMS field, got shape {data.shape} from {path}")
    return clean_mrms_values(data, product), lats, lons, valid_time


def _as_utc_iso(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, np.datetime64):
        if np.isnat(value):
            return None
        seconds = value.astype("datetime64[s]").astype(int)
        dt = datetime.fromtimestamp(int(seconds), tz=timezone.utc)
        return dt.isoformat()
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.isoformat()
    if isinstance(value, np.ndarray) and value.size == 1:
        return _as_utc_iso(value.reshape(-1)[0])
    return None


def get_valid_time(path: Path) -> str | None:
    """Extract the MRMS valid time through cfgrib/xarray, not pygrib."""
    ds = read_grib(path)
    try:
        # cfgrib normally exposes valid_time as a scalar coordinate.
        for key in ("valid_time", "time"):
            if key in ds.coords:
                result = _as_utc_iso(ds.coords[key].values)
                if result:
                    return result

        var_name = get_variable_name(ds)
        attrs = {}
        attrs.update(ds.attrs)
        attrs.update(ds[var_name].attrs)

        date_value = attrs.get("GRIB_validityDate", attrs.get("GRIB_dataDate"))
        time_value = attrs.get("GRIB_validityTime", attrs.get("GRIB_dataTime"))
        if date_value is not None and time_value is not None:
            date_text = str(int(date_value)).zfill(8)
            time_text = str(int(time_value)).zfill(6)
            dt = datetime.strptime(date_text + time_text, "%Y%m%d%H%M%S")
            return dt.replace(tzinfo=timezone.utc).isoformat()
    finally:
        ds.close()
    return None


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
            "GRIB_shortName", "GRIB_name", "GRIB_units", "GRIB_cfName",
            "GRIB_cfVarName", "GRIB_paramId", "GRIB_typeOfLevel", "GRIB_gridType",
            "GRIB_Nx", "GRIB_Ny", "GRIB_dataDate", "GRIB_dataTime",
            "GRIB_validityDate", "GRIB_validityTime",
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
