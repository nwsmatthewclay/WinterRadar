from __future__ import annotations

from pathlib import Path

import cfgrib
import numpy as np
import xarray as xr


def read_grib(path: Path) -> xr.Dataset:
    """Read the first GRIB message/group from an MRMS field."""
    datasets = cfgrib.open_datasets(str(path), backend_kwargs={"indexpath": ""})
    if not datasets:
        raise RuntimeError(f"No GRIB messages found in {path}")
    ds = datasets[0]
    return ds


def get_values(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ds = read_grib(path)
    var_name = next(v for v in ds.data_vars if v not in {"latitude", "longitude"})
    data = ds[var_name].values.astype(np.float32)
    lats = ds.latitude.values
    lons = ds.longitude.values
    return data, lats, lons
