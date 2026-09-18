from __future__ import annotations

from pathlib import Path

import cfgrib
import numpy as np
import xarray as xr

from config import field_info


def read_grib(path: Path) -> xr.Dataset:
    """Read the first GRIB message/group from an MRMS field."""
    datasets = cfgrib.open_datasets(
        str(path),
        backend_kwargs={"indexpath": ""},
    )

    if not datasets:
        raise RuntimeError(f"No GRIB messages found in {path}")

    return datasets[0]


def get_variable_name(ds: xr.Dataset) -> str:
    """Return the primary meteorological data variable name."""
    candidates = [
        name
        for name in ds.data_vars
        if name not in {"latitude", "longitude"}
    ]

    if not candidates:
        raise RuntimeError(
            "No meteorological data variable found in GRIB dataset."
        )

    return candidates[0]


def clean_mrms_values(data: np.ndarray, product: str) -> np.ndarray:
    """
    Convert product-specific MRMS fill/no-coverage values to NaN.
    """
    arr = np.asarray(data, dtype=np.float32).copy()
    info = field_info(product)

    for value in (*info["missing"], *info["no_coverage"]):
        arr[arr == value] = np.nan

    return arr


def product_from_path(path: Path) -> str:
    """Infer the MRMS product name from the standard local filename."""
    name = path.name

    prefix = "MRMS_"
    suffix = ".latest.grib2"

    if not name.startswith(prefix) or not name.endswith(suffix):
        raise ValueError(
            f"Cannot infer MRMS product from filename: {name}"
        )

    return name[len(prefix):-len(suffix)]


def get_metadata(path: Path) -> dict:
    """Return GRIB metadata plus configured MRMS metadata."""
    product = product_from_path(path)
    ds = read_grib(path)
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


def get_values(
    path: Path,
    product: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read an MRMS field and return cleaned values, latitudes, longitudes."""
    if product is None:
        product = product_from_path(path)

    ds = read_grib(path)
    var_name = get_variable_name(ds)

    data = ds[var_name].values.astype(np.float32)
    data = clean_mrms_values(data, product)

    lats = ds.latitude.values
    lons = ds.longitude.values

    return data, lats, lons
