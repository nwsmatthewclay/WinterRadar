from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import requests
import xarray as xr
from pyproj import CRS, Transformer

from config import DATA_DIR

HRRR_DIR_BASE = "https://nomads.ncep.noaa.gov/cgi-bin/filter_hrrr_2d.pl"
PROFILE_FILE = DATA_DIR / "HRRR_profile_latest.grib2"

# Viewer-relevant CONUS/Great Lakes/Northeast domain. The native MRMS radar
# remains full-domain; only this region receives the vertical phase analysis.
LEFT_LON, RIGHT_LON = -100.0, -65.0
BOTTOM_LAT, TOP_LAT = 30.0, 52.0
PRESSURE_LEVELS = (1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 725, 700, 675, 650, 625, 600, 575, 550, 525, 500)


def _valid_file_url(dt: datetime, fhr: int) -> str:
    date = dt.strftime("%Y%m%d")
    cyc = dt.strftime("%H")
    name = f"hrrr.t{cyc}z.wrfprsf{fhr:02d}.grib2"
    directory = f"/hrrr.{date}/conus"
    params = {"file": name, "dir": directory}
    for lev in PRESSURE_LEVELS:
        params[f"lev_{lev}_mb"] = "on"
    for var in ("TMP", "DPT", "HGT"):
        params[f"var_{var}"] = "on"
    params.update({
        "subregion": "",
        "leftlon": str(LEFT_LON),
        "rightlon": str(RIGHT_LON),
        "toplat": str(TOP_LAT),
        "bottomlat": str(BOTTOM_LAT),
    })
    return HRRR_DIR_BASE + "?" + urlencode(params)


def _find_available_file(target: datetime) -> tuple[datetime, int, str]:
    target = target.replace(minute=0, second=0, microsecond=0)
    session = requests.Session()
    for back in range(0, 7):
        cycle = target - timedelta(hours=back)
        fhr = 0
        # If the cycle is older than target, use the matching forecast hour.
        # Try exact-cycle first, then prior cycles with positive forecast hours.
        candidates = [(cycle, fhr)]
        if back > 0:
            candidates = [(cycle, back)]
        for cyc, fh in candidates:
            url = _valid_file_url(cyc, fh)
            try:
                r = session.get(url, stream=True, timeout=(15, 30))
                if r.status_code == 200:
                    first = next(r.iter_content(4096), b"")
                    if first.startswith(b"GRIB"):
                        r.close()
                        return cyc, fh, url
                r.close()
            except requests.RequestException:
                continue
    raise RuntimeError("No usable HRRR pressure-level subset found in the last 7 hours.")


def download_hrrr_profile(valid_time_utc: str) -> Path:
    dt = datetime.fromisoformat(valid_time_utc.replace("Z", "+00:00")).astimezone(timezone.utc)
    cycle, fhr, url = _find_available_file(dt)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = PROFILE_FILE.with_suffix(".part")
    print(f"  HRRR profile source: {cycle:%Y-%m-%d %H}Z F{fhr:02d}")
    print("  Downloading HRRR pressure-level TMP/DPT/HGT subset...")
    last_error = None
    for attempt in range(1, 4):
        try:
            with requests.get(url, stream=True, timeout=(20, 180)) as r:
                r.raise_for_status()
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(1024 * 1024):
                        if chunk:
                            f.write(chunk)
            if tmp.stat().st_size < 1024 or tmp.read_bytes()[:4] != b"GRIB":
                raise RuntimeError("Downloaded HRRR subset is not a valid GRIB2 payload.")
            tmp.replace(PROFILE_FILE)
            return PROFILE_FILE
        except Exception as exc:
            last_error = exc
            tmp.unlink(missing_ok=True)
            print(f"  HRRR download attempt {attempt}/3 failed: {exc}")
    raise RuntimeError(f"HRRR profile download failed after 3 attempts: {last_error}")


def _open_profile(path: Path) -> xr.Dataset:
    return xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {"typeOfLevel": "isobaricInhPa"},
            "indexpath": "",
            "read_keys": [
                "gridType", "LaDInDegrees", "LoVInDegrees",
                "Latin1InDegrees", "Latin2InDegrees", "DxInMetres",
                "DyInMetres", "latitudeOfFirstGridPointInDegrees",
                "longitudeOfFirstGridPointInDegrees", "Nx", "Ny",
                "iScansNegatively", "jScansPositively",
            ],
        },
    )


def load_hrrr_profile(path: Path) -> dict:
    ds = _open_profile(path)
    try:
        required = {"t", "dpt", "gh"}
        missing = required - set(ds.data_vars)
        if missing:
            raise RuntimeError(f"HRRR profile missing required fields: {sorted(missing)}")
        levels = np.asarray(ds.isobaricInhPa.values, dtype=np.float32)
        order = np.argsort(levels)[::-1]
        levels = levels[order]
        temp_c = np.asarray(ds.t.values, dtype=np.float32)[order] - 273.15
        dpt_c = np.asarray(ds.dpt.values, dtype=np.float32)[order] - 273.15
        height_m = np.asarray(ds.gh.values, dtype=np.float32)[order]
        from metpy.calc import wet_bulb_temperature
        from metpy.units import units
        pressure = levels[:, None, None] * units.hectopascal
        tw = wet_bulb_temperature(pressure, temp_c * units.degC, dpt_c * units.degC).to("degC").magnitude

        # RH with respect to ice using saturation vapor pressure over water for
        # the dewpoint and over ice for the environmental temperature.
        e_actual = 6.112 * np.exp((17.67 * dpt_c) / (dpt_c + 243.5))
        e_ice = 6.112 * np.exp((22.46 * temp_c) / (temp_c + 272.62))
        rh_ice = 100.0 * e_actual / np.maximum(e_ice, 0.01)
        rh_ice = np.clip(rh_ice, 0.0, 150.0).astype(np.float32)

        attrs = ds.t.attrs
        projection = {k: attrs[k] for k in attrs if k.startswith("GRIB_")}
        return {
            "pressure_hpa": levels,
            "wetbulb_c": np.asarray(tw, dtype=np.float32),
            "temperature_c": temp_c,
            "rh_ice_pct": rh_ice,
            "height_m": height_m,
            "latitude": np.asarray(ds.latitude.values),
            "longitude": np.asarray(ds.longitude.values),
            "projection": projection,
        }
    finally:
        ds.close()


def _transformer(projection: dict) -> tuple[Transformer, float, float, float, float]:
    if projection.get("GRIB_gridType") != "lambert":
        raise RuntimeError(f"Expected HRRR Lambert grid, got {projection.get('GRIB_gridType')}")
    crs = CRS.from_proj4(
        f"+proj=lcc +lat_1={projection['GRIB_Latin1InDegrees']} "
        f"+lat_2={projection['GRIB_Latin2InDegrees']} "
        f"+lat_0={projection['GRIB_LaDInDegrees']} "
        f"+lon_0={projection['GRIB_LoVInDegrees']} +a=6371229 +b=6371229 +units=m"
    )
    return Transformer.from_crs("EPSG:4326", crs, always_xy=True), crs


def sample_profile_to_mrms(profile: dict, lats: np.ndarray, lons: np.ndarray) -> dict:
    """Nearest-neighbor sample HRRR profile onto a chunk of MRMS lat/lon."""
    transformer, _ = _transformer(profile["projection"])
    lat = np.asarray(lats, dtype=np.float64)
    lon = np.asarray(lons, dtype=np.float64)
    if lat.ndim == 1:
        lat2 = lat[:, None]
        lon2 = lon[None, :]
    else:
        lat2, lon2 = lat, lon
    x, y = transformer.transform(lon2, lat2)

    hlat = profile["latitude"]
    hlon = np.where(profile["longitude"] > 180.0, profile["longitude"] - 360.0, profile["longitude"])
    # Derive x/y index mapping from first grid point and grid spacing.
    x0, y0 = transformer.transform(float(hlon[0, 0]), float(hlat[0, 0]))
    dx = float(profile["projection"]["GRIB_DxInMetres"])
    dy = float(profile["projection"]["GRIB_DyInMetres"])
    ix = np.rint((x - x0) / dx).astype(np.int64)
    iy = np.rint((y - y0) / dy).astype(np.int64)
    ny, nx = hlat.shape
    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)

    out = {}
    for key in ("wetbulb_c", "temperature_c", "rh_ice_pct", "height_m"):
        src = profile[key]
        sampled = src[:, iy, ix].astype(np.float32, copy=False)
        sampled[:, ~valid] = np.nan
        out[key] = sampled
    out["pressure_hpa"] = profile["pressure_hpa"]
    out["valid"] = valid
    return out
