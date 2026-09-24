from __future__ import annotations

"""Fast, robust RAP pressure-level reader for WinterRadar phase analysis.

The workflow downloads a small NOMADS RAP awp130 subset containing TMP, RH,
and HGT on pressure levels.  This module deliberately avoids opening the
entire GRIB with xarray/cfgrib or calling MetPy's iterative wet-bulb solver on
every grid point.  ecCodes reads the filtered GRIB directly in one pass, and a
vectorized psychrometric Newton solve derives wet-bulb temperature.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
import time

import numpy as np
import requests
from pyproj import CRS, Transformer

from config import DATA_DIR

RAP_FILTER_URL = "https://nomads.ncep.noaa.gov/cgi-bin/filter_rap.pl"
RAP_PROFILE_FILE = DATA_DIR / "RAP_profile_latest.grib2"

LEFT_LON, RIGHT_LON = -130.0, -60.0
BOTTOM_LAT, TOP_LAT = 20.0, 55.0
PRESSURE_LEVELS = (
    1000, 975, 950, 925, 900, 875, 850, 825, 800, 775,
    750, 725, 700, 675, 650, 625, 600, 575, 550, 525, 500,
)

TMP_NAMES = {"t", "tmp"}
RH_NAMES = {"r", "rh", "relative_humidity"}
HGT_NAMES = {"gh", "hgt", "z"}


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
        "var_TMP": "on",
        "var_RH": "on",
        "var_HGT": "on",
    }
    for lev in PRESSURE_LEVELS:
        params[f"lev_{lev}_mb"] = "on"
    return RAP_FILTER_URL + "?" + urlencode(params)


def _candidate_sequence(target: datetime):
    target_hour = target.replace(minute=0, second=0, microsecond=0)
    for back in range(0, 7):
        cycle = target_hour - timedelta(hours=back)
        yield cycle, back


def download_rap_profile(valid_time_utc: str) -> Path:
    """Download one compact RAP pressure-level subset."""
    target = datetime.fromisoformat(valid_time_utc.replace("Z", "+00:00")).astimezone(timezone.utc)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    part = RAP_PROFILE_FILE.with_suffix(".part")
    headers = {"User-Agent": "WinterRadar/phase-profile (NWS operational decision support)"}
    last_error: Exception | None = None

    for attempt_index, (cycle, fhr) in enumerate(_candidate_sequence(target), start=1):
        url = _url(cycle, fhr)
        print(
            f"  RAP profile candidate {attempt_index}/7: "
            f"{cycle:%Y-%m-%d %H}Z F{fhr:02d} "
            f"({LEFT_LON:g} to {RIGHT_LON:g}, {BOTTOM_LAT:g} to {TOP_LAT:g}; full CONUS)",
            flush=True,
        )
        try:
            with requests.get(url, stream=True, timeout=(20, 120), headers=headers) as r:
                if r.status_code == 404:
                    print("  Candidate unavailable (HTTP 404); trying the next cycle.", flush=True)
                    last_error = RuntimeError(f"HTTP 404 for RAP candidate {cycle:%Y-%m-%d %H}Z F{fhr:02d}")
                    continue
                r.raise_for_status()
                content_length = r.headers.get("Content-Length")
                expected = int(content_length) if content_length else None
                print(
                    f"  RAP response size: {expected / 1024 / 1024:.1f} MiB"
                    if expected else "  RAP response size: unknown (chunked transfer)",
                    flush=True,
                )

                total = 0
                first_chunk = True
                next_report = 256 * 1024
                with part.open("wb") as fh:
                    for chunk in r.iter_content(256 * 1024):
                        if not chunk:
                            continue
                        if first_chunk:
                            first_chunk = False
                            if not chunk.startswith(b"GRIB"):
                                raise RuntimeError("RAP filter response did not begin with a GRIB2 message.")
                        fh.write(chunk)
                        total += len(chunk)
                        if total >= next_report:
                            if expected:
                                pct = 100.0 * total / expected
                                print(f"  RAP download: {total / 1024 / 1024:.2f} MiB ({pct:.0f}%)", flush=True)
                            else:
                                print(f"  RAP download: {total / 1024 / 1024:.2f} MiB", flush=True)
                            next_report += 2 * 1024 * 1024

                if total < 4096:
                    raise RuntimeError(f"RAP subset is unexpectedly small ({total} bytes).")
                if expected is not None and total != expected:
                    raise RuntimeError(f"RAP download size mismatch: expected {expected}, got {total}")

            print(f"  RAP subset download complete: {total / 1024 / 1024:.2f} MiB", flush=True)
            part.replace(RAP_PROFILE_FILE)
            return RAP_PROFILE_FILE
        except Exception as exc:
            last_error = exc
            part.unlink(missing_ok=True)
            print(f"  RAP candidate failed: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(5.0)

    raise RuntimeError(f"No usable RAP pressure-level subset found in the last 7 hours: {last_error}")


def _get_string(gid, key: str) -> str | None:
    try:
        import eccodes
        return str(eccodes.codes_get(gid, key))
    except Exception:
        return None


def _get_float(gid, key: str) -> float | None:
    try:
        import eccodes
        value = eccodes.codes_get(gid, key)
        return float(value)
    except Exception:
        return None


def _get_int(gid, key: str) -> int | None:
    try:
        import eccodes
        value = eccodes.codes_get(gid, key)
        return int(value)
    except Exception:
        return None


def _safe_array(gid, key: str) -> np.ndarray:
    import eccodes
    return np.asarray(eccodes.codes_get_array(gid, key), dtype=np.float64)


def _projection_from_gid(gid) -> dict:
    keys = (
        "gridType",
        "LaDInDegrees",
        "LoVInDegrees",
        "Latin1InDegrees",
        "Latin2InDegrees",
        "DxInMetres",
        "DyInMetres",
        "Nx",
        "Ny",
        "iScansNegatively",
        "jScansPositively",
    )
    out: dict[str, float | int | str] = {}
    for key in keys:
        try:
            import eccodes
            value = eccodes.codes_get(gid, key)
            out[f"GRIB_{key}"] = value
        except Exception:
            continue
    return out


def _valid_time_from_gid(gid) -> str | None:
    import eccodes

    # Prefer explicit validity keys when present.
    for date_key, time_key in (("validityDate", "validityTime"),):
        try:
            date = int(eccodes.codes_get(gid, date_key))
            hhmm = int(eccodes.codes_get(gid, time_key))
            dt = datetime.strptime(f"{date:08d}{hhmm:04d}", "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
            return dt.isoformat()
        except Exception:
            pass

    try:
        date = int(eccodes.codes_get(gid, "dataDate"))
        hhmm = int(eccodes.codes_get(gid, "dataTime"))
        forecast = int(eccodes.codes_get(gid, "forecastTime"))
        dt = datetime.strptime(f"{date:08d}{hhmm:04d}", "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
        return (dt + timedelta(hours=forecast)).isoformat()
    except Exception:
        return None


def _decode_pressure_fields(path: Path) -> dict:
    """Read TMP/RH/HGT from the filtered GRIB in one sequential ecCodes pass."""
    import eccodes

    wanted = {"TMP": TMP_NAMES, "RH": RH_NAMES, "HGT": HGT_NAMES}
    fields: dict[str, dict[float, np.ndarray]] = {k: {} for k in wanted}
    lat = lon = None
    projection: dict = {}
    valid_time = None
    shape = None
    n_messages = 0

    print("  Reading RAP GRIB with ecCodes (single pass)...", flush=True)
    with path.open("rb") as fh:
        while True:
            gid = eccodes.codes_grib_new_from_file(fh)
            if gid is None:
                break
            n_messages += 1
            try:
                type_of_level = _get_string(gid, "typeOfLevel")
                if type_of_level != "isobaricInhPa":
                    continue
                short_name = (_get_string(gid, "shortName") or "").lower()
                level = _get_float(gid, "level")
                if level is None:
                    continue

                field_name = None
                for name, candidates in wanted.items():
                    if short_name in candidates:
                        field_name = name
                        break
                if field_name is None:
                    continue

                # Keep only requested pressure levels.
                nearest = min(PRESSURE_LEVELS, key=lambda x: abs(x - level))
                if abs(nearest - level) > 0.01:
                    continue

                values = _safe_array(gid, "values")
                ni = _get_int(gid, "Ni")
                nj = _get_int(gid, "Nj")
                if ni is None or nj is None or ni * nj != values.size:
                    raise RuntimeError(f"Unexpected RAP grid shape: Ni={ni}, Nj={nj}, values={values.size}")
                values = values.reshape(nj, ni)

                missing = _get_float(gid, "missingValue")
                if missing is not None and np.isfinite(missing):
                    values[np.isclose(values, missing, rtol=0.0, atol=1e-6)] = np.nan
                values[~np.isfinite(values)] = np.nan

                fields[field_name][nearest] = values
                shape = values.shape

                if lat is None:
                    lat = _safe_array(gid, "latitudes").reshape(shape)
                    lon = _safe_array(gid, "longitudes").reshape(shape)
                    lon = np.where(lon > 180.0, lon - 360.0, lon)
                    projection = _projection_from_gid(gid)
                    valid_time = _valid_time_from_gid(gid)
            finally:
                eccodes.codes_release(gid)

    print(f"  RAP GRIB messages inspected: {n_messages}", flush=True)
    if lat is None or lon is None or not shape:
        raise RuntimeError("RAP GRIB contained no usable isobaric pressure-level fields.")

    missing_fields = [name for name, data in fields.items() if len(data) != len(PRESSURE_LEVELS)]
    if missing_fields:
        details = ", ".join(f"{name}: {sorted(data)}" for name, data in fields.items())
        raise RuntimeError(f"RAP pressure-level fields incomplete ({details}); missing {missing_fields}")

    ordered = {}
    for name in wanted:
        ordered[name] = np.stack([fields[name][lev] for lev in PRESSURE_LEVELS]).astype(np.float32)

    return {
        "temperature_k": ordered["TMP"],
        "rh_pct": ordered["RH"],
        "height_m": ordered["HGT"],
        "pressure_hpa": np.asarray(PRESSURE_LEVELS, dtype=np.float32),
        "latitude": np.asarray(lat, dtype=np.float32),
        "longitude": np.asarray(lon, dtype=np.float32),
        "projection": projection,
        "valid_time_utc": valid_time,
    }


def _sat_vapor_pressure_water(temp_c: np.ndarray) -> np.ndarray:
    """Magnus saturation vapor pressure over water, hPa."""
    t = np.asarray(temp_c, dtype=np.float32)
    denom = np.maximum(t + 243.5, 0.1)
    return (6.112 * np.exp(17.67 * t / denom)).astype(np.float32)


def _dewpoint_from_rh(temp_c: np.ndarray, rh_pct: np.ndarray) -> np.ndarray:
    """Vectorized Magnus inversion for dewpoint temperature in Celsius."""
    t = np.asarray(temp_c, dtype=np.float32)
    rh = np.clip(np.asarray(rh_pct, dtype=np.float32), 0.1, 100.0)
    a = 17.67
    b = 243.5
    gamma = np.log(rh / 100.0) + (a * t) / (b + np.maximum(t, -243.0))
    return (b * gamma / np.maximum(a - gamma, 0.01)).astype(np.float32)


def _wetbulb_vectorized(temp_c: np.ndarray, dewpoint_c: np.ndarray, pressure_hpa: np.ndarray) -> np.ndarray:
    """Vectorized psychrometric wet-bulb solution using Newton iterations."""
    t = np.asarray(temp_c, dtype=np.float32)
    td = np.asarray(dewpoint_c, dtype=np.float32)
    p = np.asarray(pressure_hpa, dtype=np.float32)[:, None, None]

    es_td = _sat_vapor_pressure_water(td)
    tw = np.clip(0.5 * (t + td), td, t).astype(np.float32)

    # Psychrometric constant in hPa/K; Lv varies weakly with temperature.
    cp = 1004.0
    epsilon = 0.622
    lv = np.maximum(2.45e6 - 2360.0 * t, 2.2e6)
    gamma = (cp * p) / (epsilon * lv)

    valid = np.isfinite(t) & np.isfinite(td) & np.isfinite(p) & np.isfinite(es_td)
    tw = np.where(valid, tw, np.nan)

    for _ in range(8):
        es = _sat_vapor_pressure_water(tw)
        d_es = es * 17.67 * 243.5 / np.maximum((tw + 243.5) ** 2, 0.01)
        f = es - gamma * (t - tw) - es_td
        df = d_es + gamma
        step = np.divide(f, np.maximum(df, 1e-4), out=np.zeros_like(f), where=np.isfinite(df))
        tw = np.clip(tw - step, td, t)

    return tw.astype(np.float32)


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


def load_rap_profile(path: Path) -> dict:
    """Load RAP TMP/RH/HGT and derive DPT, wet-bulb, and RH-ice."""
    if not path.exists():
        raise FileNotFoundError(path)

    data = _decode_pressure_fields(path)
    temp_c = data["temperature_k"] - 273.15
    rh_pct = np.clip(data["rh_pct"], 0.1, 100.0).astype(np.float32)

    print(f"  RAP fields decoded: {temp_c.shape}", flush=True)
    print("  Deriving pressure-level dew point from TMP + RH...", flush=True)
    dpt_c = _dewpoint_from_rh(temp_c, rh_pct)

    print("  Deriving pressure-level wet-bulb temperature...", flush=True)
    wetbulb_c = _wetbulb_vectorized(temp_c, dpt_c, data["pressure_hpa"])

    e_actual = _sat_vapor_pressure_water(dpt_c)
    e_ice = 6.112 * np.exp((22.46 * temp_c) / np.maximum(temp_c + 272.62, 0.1))
    rh_ice = np.clip(100.0 * e_actual / np.maximum(e_ice, 0.01), 0.0, 150.0).astype(np.float32)

    # Cache projected RAP coordinates so sample_profile_to_mrms does not
    # transform the RAP grid again for every MRMS chunk.
    transformer = _make_transformer(data["projection"])

    # pyproj may interpret large 2-D coordinate arrays as mixed-dimensional
    # input and raise: "x, y, z, and time must be same size if included."
    # Flatten the paired lon/lat arrays explicitly, transform them as a simple
    # x/y vector, then restore the RAP grid shape. This is the same safe pattern
    # used when sampling the RAP grid onto MRMS below.
    rap_shape = data["longitude"].shape
    hx_flat, hy_flat = transformer.transform(
        np.asarray(data["longitude"], dtype=np.float64).ravel(),
        np.asarray(data["latitude"], dtype=np.float64).ravel(),
    )
    hx = np.asarray(hx_flat, dtype=np.float64).reshape(rap_shape)
    hy = np.asarray(hy_flat, dtype=np.float64).reshape(rap_shape)

    x_axis = hx[0, :].astype(np.float64)
    y_axis = hy[:, 0].astype(np.float64)

    profile = {
        "pressure_hpa": data["pressure_hpa"],
        "wetbulb_c": wetbulb_c,
        "temperature_c": temp_c.astype(np.float32),
        "rh_ice_pct": rh_ice,
        "height_m": data["height_m"].astype(np.float32),
        "latitude": data["latitude"],
        "longitude": data["longitude"],
        "projection": data["projection"],
        "valid_time_utc": data["valid_time_utc"],
        "_projected_x": hx.astype(np.float64),
        "_projected_y": hy.astype(np.float64),
        "_x_axis": x_axis,
        "_y_axis": y_axis,
    }

    print("  RAP PROFILE READY", flush=True)
    print(f"    Valid time: {data['valid_time_utc'] or 'unknown'}", flush=True)
    print(f"    Pressure levels: {len(data['pressure_hpa'])}", flush=True)
    print(f"    Levels: {', '.join(f'{v:g}' for v in data['pressure_hpa'])} hPa", flush=True)
    print(f"    TMP: OK ({temp_c.shape})", flush=True)
    print(f"    RH: OK ({rh_pct.shape})", flush=True)
    print(f"    DPT: DERIVED FROM TMP+RH ({dpt_c.shape})", flush=True)
    print(f"    HGT: OK ({data['height_m'].shape})", flush=True)
    print(f"    Wet-bulb: OK ({wetbulb_c.shape})", flush=True)
    return profile


def _nearest_index(sorted_values: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Return nearest indices for a monotonic coordinate axis."""
    axis = np.asarray(sorted_values, dtype=np.float64)
    vals = np.asarray(values, dtype=np.float64)
    if axis.size < 2:
        return np.zeros(vals.shape, dtype=np.int64)
    idx = np.searchsorted(axis, vals)
    idx = np.clip(idx, 1, axis.size - 1)
    left = idx - 1
    right = idx
    choose_right = np.abs(vals - axis[right]) < np.abs(vals - axis[left])
    return np.where(choose_right, right, left).astype(np.int64)


def sample_profile_to_mrms(profile: dict, lats: np.ndarray, lons: np.ndarray) -> dict:
    """Nearest-neighbor sample RAP pressure-level fields onto an MRMS chunk.

    RAP is a regular Lambert grid.  Use its projected grid spacing directly
    rather than assuming that the first row/column are perfectly represented
    by searchsorted bounds.  This is particularly important for the expanded
    full-CONUS RAP subset, where the projected coordinate arrays are large and
    the outer MRMS grid extends beyond the RAP subset.
    """
    lat = np.asarray(lats, dtype=np.float64)
    lon = np.asarray(lons, dtype=np.float64)
    if lat.ndim == 1 and lon.ndim == 1:
        lat2, lon2 = np.broadcast_arrays(lat[:, None], lon[None, :])
    elif lat.ndim == 2 and lon.ndim == 2:
        if lat.shape != lon.shape:
            raise RuntimeError(f"RAP sampling coordinate shape mismatch: lat={lat.shape}, lon={lon.shape}")
        lat2, lon2 = lat, lon
    else:
        lat2, lon2 = np.broadcast_arrays(lat, lon)

    transformer = _make_transformer(profile["projection"])
    hx = np.asarray(profile.get("_projected_x"), dtype=np.float64)
    hy = np.asarray(profile.get("_projected_y"), dtype=np.float64)
    x_axis = np.asarray(profile.get("_x_axis"), dtype=np.float64)
    y_axis = np.asarray(profile.get("_y_axis"), dtype=np.float64)

    if hx.ndim != 2 or hy.ndim != 2:
        raise RuntimeError(f"RAP projected coordinate arrays must be 2-D: x={hx.shape}, y={hy.shape}")
    if x_axis.size != hx.shape[1] or y_axis.size != hy.shape[0]:
        raise RuntimeError(
            f"RAP projected axis mismatch: x_axis={x_axis.shape}, y_axis={y_axis.shape}, grid={hx.shape}"
        )

    # The RAP Lambert grid is regular in projected x/y.  Derive the spacing
    # from the full axes and use the actual grid origin, preserving scan
    # direction.  This avoids the zero-valid-cell failure that can occur when
    # projected bounds are compared against a malformed/reversed axis.
    dx_values = np.diff(x_axis)
    dy_values = np.diff(y_axis)
    dx = float(np.nanmedian(dx_values[np.isfinite(dx_values) & (np.abs(dx_values) > 0)]))
    dy = float(np.nanmedian(dy_values[np.isfinite(dy_values) & (np.abs(dy_values) > 0)]))
    if not np.isfinite(dx) or not np.isfinite(dy):
        raise RuntimeError("RAP projected grid spacing could not be determined")

    x_flat, y_flat = transformer.transform(lon2.ravel(), lat2.ravel())
    x = np.asarray(x_flat, dtype=np.float64).reshape(lat2.shape)
    y = np.asarray(y_flat, dtype=np.float64).reshape(lat2.shape)

    x0 = float(x_axis[0])
    y0 = float(y_axis[0])
    ix = np.rint((x - x0) / dx).astype(np.int64)
    iy = np.rint((y - y0) / dy).astype(np.int64)

    # Accept a target point when its projected coordinate is within half a RAP
    # grid cell of the available subset.  This is a geometric validity test,
    # not merely an array-index clipping test.
    half_dx = abs(dx) * 0.51
    half_dy = abs(dy) * 0.51
    x_min = min(float(x_axis.min()), float(x_axis.max())) - half_dx
    x_max = max(float(x_axis.min()), float(x_axis.max())) + half_dx
    y_min = min(float(y_axis.min()), float(y_axis.max())) - half_dy
    y_max = max(float(y_axis.min()), float(y_axis.max())) + half_dy
    valid = (
        np.isfinite(x) & np.isfinite(y)
        & (x >= x_min) & (x <= x_max)
        & (y >= y_min) & (y <= y_max)
        & (ix >= 0) & (ix < x_axis.size)
        & (iy >= 0) & (iy < y_axis.size)
    )

    # Keep invalid indices safe for NumPy advanced indexing; invalid samples
    # are masked immediately afterward.
    ix_safe = np.clip(ix, 0, x_axis.size - 1)
    iy_safe = np.clip(iy, 0, y_axis.size - 1)

    out: dict[str, np.ndarray] = {}
    for key in ("wetbulb_c", "temperature_c", "rh_ice_pct", "height_m"):
        src = np.asarray(profile[key])
        sampled = src[:, iy_safe, ix_safe].astype(np.float32, copy=False)
        sampled[:, ~valid] = np.nan
        out[key] = sampled

    out["pressure_hpa"] = np.asarray(profile["pressure_hpa"], dtype=np.float32)
    out["valid"] = valid
    return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Download and validate a RAP winter-phase profile.")
    parser.add_argument("--valid-time", required=True, help="ISO UTC valid time")
    args = parser.parse_args()
    path = download_rap_profile(args.valid_time)
    profile = load_rap_profile(path)
    print(f"RAP PROFILE READY: {path}")
    print(f"RAP levels loaded: {len(profile['pressure_hpa'])}")


if __name__ == "__main__":
    main()
