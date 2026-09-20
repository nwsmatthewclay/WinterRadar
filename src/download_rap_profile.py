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
    if not path.exists():
        raise FileNotFoundError(path)
    import cfgrib
    return cfgrib.open_datasets(str(path), backend_kwargs={"indexpath": ""})


def _find_dataset(datasets: list[xr.Dataset], predicate):
    for ds in datasets:
        if predicate(ds):
            return ds
    return None


def _pick_var(ds: xr.Dataset, names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in ds.data_vars:
            return name
    return None


def _pressure_projection(ds: xr.Dataset) -> dict:
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


def load_rap_profile(path: Path) -> dict:
    datasets = _open_datasets(path)
    try:
        pds = _find_dataset(
            datasets,
            lambda ds: "isobaricInhPa" in ds.coords
            and _pick_var(ds, ("t", "tmp")) is not None
            and _pick_var(ds, ("dpt", "td", "2d")) is not None
            and _pick_var(ds, ("gh", "hgt", "z")) is not None,
        )
        if pds is None:
            raise RuntimeError("RAP pressure-level dataset with TMP/DPT/HGT was not found.")

        tname = _pick_var(pds, ("t", "tmp"))
        dname = _pick_var(pds, ("dpt", "td", "2d"))
        zname = _pick_var(pds, ("gh", "hgt", "z"))
        assert tname and dname and zname

        levels = np.asarray(pds.isobaricInhPa.values, dtype=np.float32)
        order = np.argsort(levels)[::-1]  # pressure descending -> low altitude first
        levels = levels[order]
        temp_c = np.asarray(pds[tname].values, dtype=np.float32)[order] - 273.15
        dpt_c = np.asarray(pds[dname].values, dtype=np.float32)[order] - 273.15
        height_m = np.asarray(pds[zname].values, dtype=np.float32)[order]

        from metpy.calc import wet_bulb_temperature
        from metpy.units import units
        wetbulb_c = wet_bulb_temperature(
            levels[:, None, None] * units.hectopascal,
            temp_c * units.degC,
            dpt_c * units.degC,
        ).to("degC").magnitude.astype(np.float32)

        e_actual = 6.112 * np.exp((17.67 * dpt_c) / np.maximum(dpt_c + 243.5, 0.1))
        e_ice = 6.112 * np.exp((22.46 * temp_c) / np.maximum(temp_c + 272.62, 0.1))
        rh_ice = np.clip(100.0 * e_actual / np.maximum(e_ice, 0.01), 0.0, 150.0).astype(np.float32)

        lat = np.asarray(pds.latitude.values)
        lon = np.where(np.asarray(pds.longitude.values) > 180.0,
                       np.asarray(pds.longitude.values) - 360.0,
                       np.asarray(pds.longitude.values))
        projection = _pressure_projection(pds)

        result = {
            "pressure_hpa": levels,
            "wetbulb_c": wetbulb_c,
            "temperature_c": temp_c.astype(np.float32),
            "rh_ice_pct": rh_ice,
            "height_m": height_m.astype(np.float32),
            "latitude": lat,
            "longitude": lon,
            "projection": projection,
            "valid_time_utc": _dataset_valid_time(pds),
        }

        # Optional surface/2-m anchors are retained for the next phase-engine
        # refinement and for QC, even though the current stage still uses the
        # pressure-level profile itself.
        sds = _find_dataset(
            datasets,
            lambda ds: any(c in ds.coords for c in ("surface", "heightAboveGround")),
        )
        if sds is not None:
            pvar = _pick_var(sds, ("sp", "pres", "pres_surface"))
            t2 = _pick_var(sds, ("t2m", "t2m"))
            d2 = _pick_var(sds, ("d2m", "dpt2m"))
            if pvar:
                p = np.asarray(sds[pvar].values, dtype=np.float32)
                if np.nanmedian(p) > 2000.0:
                    p = p / 100.0
                result["surface_pressure_hpa"] = p
            if t2:
                result["surface_temperature_c"] = np.asarray(sds[t2].values, dtype=np.float32) - 273.15
            if d2:
                result["surface_dewpoint_c"] = np.asarray(sds[d2].values, dtype=np.float32) - 273.15
            hgt_name = _pick_var(sds, ("orog", "gh", "hgt"))
            if hgt_name:
                result["surface_height_m"] = np.asarray(sds[hgt_name].values, dtype=np.float32)

        return result
    finally:
        for ds in datasets:
            ds.close()


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
