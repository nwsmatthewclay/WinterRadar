from __future__ import annotations

"""Retrieve a compact RAP pressure-level profile for winter precipitation type.

NOAA/NCEP explicitly recommends RAP for upper-level analysis and short-range
forecasting because the distributed HRRR product does not provide full 3-D
upper-level fields. This module uses the NOMADS RAP GRIB filter to request only
pressure-level TMP/DPT/HGT plus a few surface fields over the WinterRadar domain.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
import time

import numpy as np
import requests
import xarray as xr
from pyproj import CRS, Transformer

from config import DATA_DIR

RAP_FILTER_URL = "https://nomads.ncep.noaa.gov/cgi-bin/filter_rap.pl"
RAP_PROFILE_FILE = DATA_DIR / "RAP_profile_latest.grib2"

LEFT_LON, RIGHT_LON = -100.0, -65.0
BOTTOM_LAT, TOP_LAT = 30.0, 52.0
PRESSURE_LEVELS = (
    1000, 975, 950, 925, 900, 875, 850, 825, 800, 775,
    750, 725, 700, 675, 650, 625, 600, 575, 550, 525, 500,
)


def _url(cycle: datetime, fhr: int) -> str:
    date = cycle.strftime("%Y%m%d")
    hh = cycle.strftime("%H")
    filename = f"rap.t{hh}z.awp130pgrbf{fhr:02d}.grib2"
    params: dict[str, str] = {
        "file": filename,
        "subregion": "",
        "leftlon": str(LEFT_LON),
        "rightlon": str(RIGHT_LON),
        "toplat": str(TOP_LAT),
        "bottomlat": str(BOTTOM_LAT),
        "dir": f"/rap.{date}",
        # Surface anchors: needed later to avoid treating below-ground 1000/975-mb
        # levels as part of the surface-based thermal layer in elevated terrain.
        "lev_surface": "on",
        "lev_2_m_above_ground": "on",
        "var_PRES": "on",
        "var_TMP": "on",
        "var_DPT": "on",
        "var_HGT": "on",
    }
    for lev in PRESSURE_LEVELS:
        params[f"lev_{lev}_mb"] = "on"
    return RAP_FILTER_URL + "?" + urlencode(params)


def _response_is_grib(response: requests.Response) -> bool:
    first = next(response.iter_content(4096), b"")
    return first.startswith(b"GRIB")


def _candidate_sequence(target: datetime):
    """Yield candidate RAP cycle/fhr pairs ordered from most synchronous to older."""
    target_hour = target.replace(minute=0, second=0, microsecond=0)
    # First try the analysis at the target hour. If not yet posted, walk back
    # through prior hourly cycles using the forecast hour needed to hit the same
    # valid hour. This keeps the profile close to the MRMS valid time.
    for back in range(0, 7):
        cycle = target_hour - timedelta(hours=back)
        fhr = back
        yield cycle, fhr


def _find_available(target: datetime) -> tuple[datetime, int, str]:
    session = requests.Session()
    headers = {"User-Agent": "WinterRadar/phase-profile (NWS operational decision support)"}
    for i, (cycle, fhr) in enumerate(_candidate_sequence(target)):
        url = _url(cycle, fhr)
        try:
            with session.get(url, stream=True, timeout=(20, 30), headers=headers) as r:
                if r.status_code == 200 and _response_is_grib(r):
                    return cycle, fhr, url
        except requests.RequestException as exc:
            print(f"  RAP availability check failed for {cycle:%Y-%m-%d %H}Z F{fhr:02d}: {exc}")
        if i < 6:
            # NOMADS requests that clients pause between repeated grib-filter
            # submissions. Keep this short enough for the five-minute workflow.
            time.sleep(2.0)
    raise RuntimeError("No usable RAP pressure-level subset found in the last 7 hours.")


def download_rap_profile(valid_time_utc: str) -> Path:
    target = datetime.fromisoformat(valid_time_utc.replace("Z", "+00:00")).astimezone(timezone.utc)
    cycle, fhr, url = _find_available(target)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    part = RAP_PROFILE_FILE.with_suffix(".part")
    headers = {"User-Agent": "WinterRadar/phase-profile (NWS operational decision support)"}

    print(f"  RAP profile source: {cycle:%Y-%m-%d %H}Z F{fhr:02d} (valid ~{target:%Y-%m-%d %H:%M}Z)")
    print("  Downloading RAP pressure-level TMP/DPT/HGT subset...")

    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            with requests.get(url, stream=True, timeout=(20, 180), headers=headers) as r:
                r.raise_for_status()
                content_length = r.headers.get("Content-Length")
                with part.open("wb") as fh:
                    for chunk in r.iter_content(1024 * 1024):
                        if chunk:
                            fh.write(chunk)

                if content_length is not None and part.stat().st_size != int(content_length):
                    raise RuntimeError(
                        f"RAP download size mismatch: expected {content_length}, got {part.stat().st_size}"
                    )

            if part.stat().st_size < 4096:
                raise RuntimeError("RAP subset is unexpectedly small.")
            if part.read_bytes()[:4] != b"GRIB":
                raise RuntimeError("Downloaded RAP subset is not a GRIB2 payload.")

            # Force cfgrib/eccodes to open the file before accepting it. This also
            # prevents a server-side HTML/error body from entering the phase engine.
            import cfgrib
            groups = cfgrib.open_datasets(str(part), backend_kwargs={"indexpath": ""})
            try:
                if not groups or not any("latitude" in ds.coords and "longitude" in ds.coords for ds in groups):
                    raise RuntimeError("RAP subset lacks latitude/longitude coordinates.")
            finally:
                for ds in groups:
                    ds.close()

            part.replace(RAP_PROFILE_FILE)
            return RAP_PROFILE_FILE
        except Exception as exc:
            last_error = exc
            part.unlink(missing_ok=True)
            print(f"  RAP download attempt {attempt}/3 failed: {type(exc).__name__}: {exc}")
            if attempt < 3:
                time.sleep(2.0)

    raise RuntimeError(f"RAP profile download failed after 3 attempts: {last_error}")


def _open_datasets(path: Path) -> list[xr.Dataset]:
    """Open all useful GRIB groups without requiring TMP/DPT/HGT to share one group."""
    if not path.exists():
        raise FileNotFoundError(path)
    import cfgrib
    return cfgrib.open_datasets(str(path), backend_kwargs={"indexpath": ""})


def _open_pressure_variable(path: Path, short_names: tuple[str, ...]) -> xr.Dataset:
    """Open one pressure-level variable directly from the RAP GRIB file."""
    import cfgrib

    last_error: Exception | None = None
    for short_name in short_names:
        try:
            ds = xr.open_dataset(
                path,
                engine="cfgrib",
                backend_kwargs={
                    "indexpath": "",
                    "filter_by_keys": {
                        "typeOfLevel": "isobaricInhPa",
                        "shortName": short_name,
                    },
                },
            )
            if "isobaricInhPa" in ds.coords and ds.data_vars:
                return ds
            ds.close()
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Could not open RAP pressure-level variable {short_names}; last error: {last_error}"
    )


def _find_pressure_variable(path: Path, names: tuple[str, ...]) -> tuple[xr.Dataset, str]:
    """Find a pressure-level variable by GRIB shortName or xarray variable name."""
    import cfgrib

    # First try exact GRIB shortName filters. This is more reliable than relying
    # on cfgrib.open_datasets() to merge unrelated variables into one Dataset.
    for name in names:
        try:
            ds = xr.open_dataset(
                path,
                engine="cfgrib",
                backend_kwargs={
                    "indexpath": "",
                    "filter_by_keys": {
                        "typeOfLevel": "isobaricInhPa",
                        "shortName": name,
                    },
                },
            )
            if "isobaricInhPa" in ds.coords and ds.data_vars:
                return ds, next(iter(ds.data_vars))
            ds.close()
        except Exception:
            pass

    # Fall back to scanning cfgrib's groups for installations where shortName
    # is exposed differently.
    groups = cfgrib.open_datasets(path, backend_kwargs={"indexpath": ""})
    try:
        wanted = set(names)
        for ds in groups:
            if "isobaricInhPa" not in ds.coords:
                continue
            for var in ds.data_vars:
                short_name = str(ds[var].attrs.get("GRIB_shortName", ""))
                if var in wanted or short_name in wanted:
                    return ds, var
    finally:
        # Do not close the returned Dataset. Its caller owns it. Close only
        # datasets that were not returned.
        pass

    for ds in groups:
        try:
            ds.close()
        except Exception:
            pass
    raise RuntimeError(f"RAP pressure-level variable not found: {names}")


def _pressure_projection_from(ds: xr.Dataset) -> dict:
    attrs: dict = {}
    for name in ds.data_vars:
        attrs.update({k: v for k, v in ds[name].attrs.items() if k.startswith("GRIB_")})
    attrs.update({k: v for k, v in ds.attrs.items() if k.startswith("GRIB_")})
    return attrs


def _dataset_valid_time(ds: xr.Dataset) -> str | None:
    for coord_name in ("valid_time", "time"):
        if coord_name in ds.coords:
            arr = np.asarray(ds[coord_name].values)
            if arr.size:
                value = arr.reshape(-1)[0]
                try:
                    text = np.datetime_as_string(value.astype("datetime64[s]"), unit="s")
                    if text != "NaT":
                        return text + "+00:00"
                except (TypeError, ValueError, AttributeError):
                    pass
    return None


def _as_level_first(data: np.ndarray, nlevels: int) -> np.ndarray:
    """Return a field as [level, y, x]."""
    arr = np.asarray(data, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise RuntimeError(f"Expected 3-D pressure-level field, got shape {arr.shape}")
    if arr.shape[0] == nlevels:
        return arr
    for axis in range(1, 3):
        if arr.shape[axis] == nlevels:
            return np.moveaxis(arr, axis, 0)
    raise RuntimeError(f"Could not identify pressure dimension in field shape {arr.shape}; levels={nlevels}")


def _read_pressure_field(path: Path, names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, str]:
    ds, var = _find_pressure_variable(path, names)
    try:
        levels = np.asarray(ds["isobaricInhPa"].values, dtype=np.float32)
        order = np.argsort(levels)[::-1]
        levels = levels[order]
        data = _as_level_first(ds[var].values, len(levels))[order]
        lat = np.asarray(ds.latitude.values)
        lon = np.where(np.asarray(ds.longitude.values) > 180.0,
                       np.asarray(ds.longitude.values) - 360.0,
                       np.asarray(ds.longitude.values))
        projection = _pressure_projection_from(ds)
        valid_time = _dataset_valid_time(ds)
        return data, levels, lat, lon, projection, valid_time
    finally:
        ds.close()


def load_rap_profile(path: Path) -> dict:
    """Load RAP TMP, DPT and HGT independently from the pressure-level GRIB."""
    if not path.exists():
        raise FileNotFoundError(path)

    # RAP commonly exposes these as t, dpt, and gh. Some GRIB inventories use
    # tmp/2d/hgt, so the fallback names are retained.
    temp_c_k, levels_t, lat, lon, projection, valid_time = _read_pressure_field(path, ("t", "tmp"))
    dpt_c_k, levels_d, _, _, _, _ = _read_pressure_field(path, ("dpt", "td", "2d"))
    height_m, levels_z, _, _, _, _ = _read_pressure_field(path, ("gh", "hgt", "z"))

    if not np.array_equal(levels_t, levels_d) or not np.array_equal(levels_t, levels_z):
        raise RuntimeError(
            f"RAP pressure levels differ between TMP/DPT/HGT: "
            f"TMP={levels_t.tolist()} DPT={levels_d.tolist()} HGT={levels_z.tolist()}"
        )

    temp_c = temp_c_k - 273.15
    dpt_c = dpt_c_k - 273.15

    # Compute wet-bulb one pressure level at a time. This avoids relying on
    # MetPy broadcasting behavior across the full 3-D grid and is considerably
    # easier on GitHub Actions memory.
    from metpy.calc import wet_bulb_temperature
    from metpy.units import units

    wetbulb_c = np.empty_like(temp_c, dtype=np.float32)
    for i, pressure in enumerate(levels_t):
        wb = wet_bulb_temperature(
            pressure * units.hectopascal,
            temp_c[i] * units.degC,
            dpt_c[i] * units.degC,
        ).to("degC").magnitude
        wetbulb_c[i] = np.asarray(wb, dtype=np.float32)

    # Ice-relative humidity diagnostic. Use a stable Magnus-style formulation
    # and retain it as a supporting field rather than a hard classification.
    e_actual = 6.112 * np.exp((17.67 * dpt_c) / np.maximum(dpt_c + 243.5, 0.1))
    e_ice = 6.112 * np.exp((22.46 * temp_c) / np.maximum(temp_c + 272.62, 0.1))
    rh_ice = np.clip(100.0 * e_actual / np.maximum(e_ice, 0.01), 0.0, 150.0).astype(np.float32)

    print("  RAP PROFILE READY")
    print(f"    Valid time: {valid_time or 'unknown'}")
    print(f"    Pressure levels: {len(levels_t)}")
    print(f"    Levels: {', '.join(f'{v:g}' for v in levels_t)} hPa")
    print(f"    TMP: OK ({temp_c.shape})")
    print(f"    DPT: OK ({dpt_c.shape})")
    print(f"    HGT: OK ({height_m.shape})")
    print(f"    Wet-bulb: OK ({wetbulb_c.shape})")

    return {
        "pressure_hpa": levels_t,
        "wetbulb_c": wetbulb_c,
        "temperature_c": temp_c.astype(np.float32),
        "rh_ice_pct": rh_ice,
        "height_m": height_m.astype(np.float32),
        "latitude": lat,
        "longitude": lon,
        "projection": projection,
        "valid_time_utc": valid_time,
    }

def _nearest_index(sorted_values: np.ndarray, values: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(sorted_values, values)
    idx = np.clip(idx, 1, len(sorted_values) - 1)
    left = idx - 1
    right = idx
    choose_right = np.abs(values - sorted_values[right]) < np.abs(values - sorted_values[left])
    return np.where(choose_right, right, left).astype(np.int64)


def _make_transformer(projection: dict) -> Transformer:
    grid_type = projection.get("GRIB_gridType")
    if grid_type != "lambert":
        raise RuntimeError(f"Expected Lambert RAP grid, got {grid_type!r}")
    lat_0 = projection.get("GRIB_LaDInDegrees")
    lon_0 = projection.get("GRIB_LoVInDegrees")
    lat_1 = projection.get("GRIB_Latin1InDegrees")
    lat_2 = projection.get("GRIB_Latin2InDegrees")
    if None in (lat_0, lon_0, lat_1, lat_2):
        raise RuntimeError("RAP Lambert projection metadata is incomplete.")
    crs = CRS.from_proj4(
        f"+proj=lcc +lat_1={lat_1} +lat_2={lat_2} +lat_0={lat_0} +lon_0={lon_0} "
        "+a=6371229 +b=6371229 +units=m"
    )
    return Transformer.from_crs("EPSG:4326", crs, always_xy=True)


def sample_profile_to_mrms(profile: dict, lats: np.ndarray, lons: np.ndarray) -> dict:
    """Nearest-neighbor sample RAP profile fields onto an MRMS chunk."""
    lat = np.asarray(lats, dtype=np.float64)
    lon = np.asarray(lons, dtype=np.float64)
    if lat.ndim == 1:
        lat2, lon2 = lat[:, None], lon[None, :]
    else:
        lat2, lon2 = lat, lon

    transformer = _make_transformer(profile["projection"])
    hlat = np.asarray(profile["latitude"], dtype=np.float64)
    hlon = np.asarray(profile["longitude"], dtype=np.float64)
    hx, hy = transformer.transform(hlon, hlat)

    # RAP is a regular Lambert grid. First row/column give the 1-D projected
    # coordinate axes; use their actual orientation rather than assuming a sign.
    x_axis = hx[0, :]
    y_axis = hy[:, 0]
    x_rev = x_axis[0] > x_axis[-1]
    y_rev = y_axis[0] > y_axis[-1]
    if x_rev:
        x_sorted = x_axis[::-1]
    else:
        x_sorted = x_axis
    if y_rev:
        y_sorted = y_axis[::-1]
    else:
        y_sorted = y_axis

    x, y = transformer.transform(lon2, lat2)
    ix_sorted = _nearest_index(x_sorted, x)
    iy_sorted = _nearest_index(y_sorted, y)
    ix = (len(x_axis) - 1 - ix_sorted) if x_rev else ix_sorted
    iy = (len(y_axis) - 1 - iy_sorted) if y_rev else iy_sorted

    valid = (
        np.isfinite(x) & np.isfinite(y)
        & (x >= min(x_axis.min(), x_axis.max())) & (x <= max(x_axis.min(), x_axis.max()))
        & (y >= min(y_axis.min(), y_axis.max())) & (y <= max(y_axis.min(), y_axis.max()))
    )

    out: dict[str, np.ndarray] = {}
    for key in ("wetbulb_c", "temperature_c", "rh_ice_pct", "height_m"):
        src = np.asarray(profile[key])
        sampled = src[:, iy, ix].astype(np.float32, copy=False)
        sampled[:, ~valid] = np.nan
        out[key] = sampled

    for key in ("surface_pressure_hpa", "surface_temperature_c", "surface_dewpoint_c", "surface_height_m"):
        if key in profile:
            src = np.asarray(profile[key])
            sampled = src[iy, ix].astype(np.float32, copy=False)
            sampled[~valid] = np.nan
            out[key] = sampled

    out["pressure_hpa"] = np.asarray(profile["pressure_hpa"], dtype=np.float32)
    out["valid"] = valid
    return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Download and validate a RAP winter-phase profile.")
    parser.add_argument("--valid-time", required=True, help="ISO UTC valid time, e.g. 2026-09-20T00:42:00+00:00")
    args = parser.parse_args()
    path = download_rap_profile(args.valid_time)
    print(f"RAP PROFILE READY: {path}")


if __name__ == "__main__":
    main()
