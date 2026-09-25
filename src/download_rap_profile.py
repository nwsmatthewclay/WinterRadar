from __future__ import annotations

"""Fast, robust RAP pressure-level reader for WinterRadar phase analysis.

The workflow downloads a small NOMADS RAP awp130 subset containing TMP, RH,
and HGT on pressure levels.  This module deliberately avoids importing the
Python eccodes binding into the core process because the core MRMS reader uses
the system-linked pygrib binding.  Mixing those two native ecCodes bindings in
one Python process can trigger ``double free or corruption`` on Linux CI.
A vectorized psychrometric Newton solve derives wet-bulb temperature.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
import time

import numpy as np
import requests

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


def _grb_get(grb, key, default=None):
    try:
        return grb[key]
    except Exception:
        return default


def _projection_from_grb(grb) -> dict:
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
        "latitudeOfFirstGridPointInDegrees",
        "longitudeOfFirstGridPointInDegrees",
    )
    out: dict[str, float | int | str] = {}
    for key in keys:
        value = _grb_get(grb, key)
        if value is not None:
            out[f"GRIB_{key}"] = value
    return out


def _valid_time_from_grb(grb) -> str | None:
    for date_key, time_key in (("validityDate", "validityTime"),):
        try:
            date = int(grb[date_key])
            hhmm = int(grb[time_key])
            dt = datetime.strptime(
                f"{date:08d}{hhmm:04d}", "%Y%m%d%H%M"
            ).replace(tzinfo=timezone.utc)
            return dt.isoformat()
        except Exception:
            pass

    try:
        date = int(grb["dataDate"])
        hhmm = int(grb["dataTime"])
        forecast = int(grb["forecastTime"])
        dt = datetime.strptime(
            f"{date:08d}{hhmm:04d}", "%Y%m%d%H%M"
        ).replace(tzinfo=timezone.utc)
        return (dt + timedelta(hours=forecast)).isoformat()
    except Exception:
        return None


def _decode_pressure_fields(path: Path) -> dict:
    """Read TMP/RH/HGT from the filtered GRIB with one pygrib handle."""
    import pygrib

    wanted = {"TMP": TMP_NAMES, "RH": RH_NAMES, "HGT": HGT_NAMES}
    fields: dict[str, dict[float, np.ndarray]] = {k: {} for k in wanted}
    lat = lon = None
    projection: dict = {}
    valid_time = None
    shape = None
    n_messages = 0

    print("  Reading RAP GRIB with pygrib (single pass)...", flush=True)
    grbs = None
    try:
        grbs = pygrib.open(str(path))
        for grb in grbs:
            n_messages += 1
            type_of_level = str(_grb_get(grb, "typeOfLevel", "")).lower()
            if type_of_level != "isobaricinhpa":
                continue

            short_name = str(_grb_get(grb, "shortName", "")).lower()
            level = _grb_get(grb, "level")
            if level is None:
                continue

            field_name = None
            for name, candidates in wanted.items():
                if short_name in candidates:
                    field_name = name
                    break
            if field_name is None:
                continue

            nearest = min(PRESSURE_LEVELS, key=lambda x: abs(x - float(level)))
            if abs(nearest - float(level)) > 0.01:
                continue

            values = np.asarray(grb.values, dtype=np.float32).copy()
            if values.ndim != 2:
                raise RuntimeError(
                    f"Unexpected RAP field shape for {field_name} {level}: {values.shape}"
                )

            missing = _grb_get(grb, "missingValue")
            if missing is not None and np.isfinite(float(missing)):
                values[np.isclose(values, float(missing), rtol=0.0, atol=1e-6)] = np.nan
            values[~np.isfinite(values)] = np.nan

            fields[field_name][nearest] = values
            shape = values.shape

            if lat is None:
                projection = _projection_from_grb(grb)
                valid_time = _valid_time_from_grb(grb)

                raw_lat, raw_lon = grb.latlons()
                raw_lat = np.asarray(raw_lat, dtype=np.float64)
                raw_lon = np.asarray(raw_lon, dtype=np.float64)

                if raw_lat.shape == shape and raw_lon.shape == shape:
                    lat = raw_lat.copy()
                    lon = raw_lon.copy()
                else:
                    print(
                        f"  RAP coordinate arrays are non-native: lat={raw_lat.shape}, "
                        f"lon={raw_lon.shape}; reconstructing {shape} Lambert grid.",
                        flush=True,
                    )
                    lat, lon, _, _ = _build_regular_lambert_grid(shape, projection)

                lon = np.where(lon > 180.0, lon - 360.0, lon)
    finally:
        if grbs is not None:
            grbs.close()

    print(f"  RAP GRIB messages inspected: {n_messages}", flush=True)
    if lat is None or lon is None or not shape:
        raise RuntimeError("RAP GRIB contained no usable isobaric pressure-level fields.")

    missing_fields = [name for name, data in fields.items() if len(data) != len(PRESSURE_LEVELS)]
    if missing_fields:
        details = ", ".join(f"{name}: {sorted(data)}" for name, data in fields.items())
        raise RuntimeError(
            f"RAP pressure-level fields incomplete ({details}); missing {missing_fields}"
        )

    ordered = {}
    for name in wanted:
        ordered[name] = np.stack(
            [fields[name][lev] for lev in PRESSURE_LEVELS]
        ).astype(np.float32)

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


def _lambert_conformal_conic_xy(
    longitude_deg: np.ndarray,
    latitude_deg: np.ndarray,
    projection: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Project lon/lat to the spherical Lambert grid used by RAP.

    This intentionally implements the small forward LCC calculation locally
    instead of relying on pyproj's Transformer.transform().  Some pyproj/PROJ
    builds have intermittently interpreted large NumPy coordinate arrays as
    mixed-dimensional input and raised:
        "x, y, z, and time must be same size if included."
    The RAP grid is a regular spherical Lambert grid, so the forward equation
    is deterministic and avoids that failure mode entirely.
    """
    grid_type = str(projection.get("GRIB_gridType", "")).lower()
    if grid_type != "lambert":
        raise RuntimeError(f"Expected Lambert RAP grid, got {grid_type!r}")

    lat_0 = projection.get("GRIB_LaDInDegrees")
    lon_0 = projection.get("GRIB_LoVInDegrees")
    lat_1 = projection.get("GRIB_Latin1InDegrees")
    lat_2 = projection.get("GRIB_Latin2InDegrees")
    if None in (lat_0, lon_0, lat_1, lat_2):
        raise RuntimeError("RAP Lambert projection metadata is incomplete.")

    lon = np.asarray(longitude_deg, dtype=np.float64)
    lat = np.asarray(latitude_deg, dtype=np.float64)
    if lon.shape != lat.shape:
        raise RuntimeError(
            f"RAP projection coordinate shape mismatch: lon={lon.shape}, lat={lat.shape}"
        )

    # RAP's Lambert grid uses a spherical Earth with radius 6371229 m.
    radius = 6371229.0
    phi0 = np.deg2rad(float(lat_0))
    phi1 = np.deg2rad(float(lat_1))
    phi2 = np.deg2rad(float(lat_2))
    lam0 = np.deg2rad(float(lon_0))

    def _t(phi):
        # tan(pi/4 + phi/2) is well behaved for the RAP CONUS domain.
        return np.tan(np.pi / 4.0 + phi / 2.0)

    if abs(float(lat_1) - float(lat_2)) < 1.0e-8:
        n = np.sin(phi1)
    else:
        n = np.log(np.cos(phi1) / np.cos(phi2)) / np.log(_t(phi2) / _t(phi1))

    if not np.isfinite(n) or abs(n) < 1.0e-12:
        raise RuntimeError(f"Invalid RAP Lambert projection exponent: {n!r}")

    f = np.cos(phi1) * (_t(phi1) ** n) / n
    rho0 = radius * f / (_t(phi0) ** n)

    phi = np.deg2rad(lat)
    lam = np.deg2rad(lon)
    rho = radius * f / (_t(phi) ** n)
    theta = n * (lam - lam0)

    x = rho * np.sin(theta)
    y = rho0 - rho * np.cos(theta)
    return x, y



def _build_regular_lambert_grid(shape: tuple[int, int], projection: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build RAP grid coordinates from GRIB geometry when coordinate arrays are malformed.

    NOMADS/ecCodes can occasionally expose the filtered RAP latitude/longitude
    arrays with a reduced or broadcast shape even though the decoded field is
    a regular Lambert grid.  The GRIB projection metadata contains everything
    needed to reconstruct the exact regular grid, so use that as the
    authoritative geometry rather than trusting a malformed coordinate array.
    """
    ny, nx = int(shape[0]), int(shape[1])
    dx = projection.get("GRIB_DxInMetres")
    dy = projection.get("GRIB_DyInMetres")
    first_lat = projection.get("GRIB_latitudeOfFirstGridPointInDegrees")
    first_lon = projection.get("GRIB_longitudeOfFirstGridPointInDegrees")
    if None in (dx, dy, first_lat, first_lon):
        raise RuntimeError(
            "RAP Lambert grid geometry is incomplete: "
            f"Dx={dx!r}, Dy={dy!r}, first_lat={first_lat!r}, first_lon={first_lon!r}"
        )

    # Project the first grid point using the same spherical LCC equations used
    # for all subsequent sampling.
    first_x, first_y = _lambert_conformal_conic_xy(
        np.asarray([float(first_lon)], dtype=np.float64),
        np.asarray([float(first_lat)], dtype=np.float64),
        projection,
    )
    x0 = float(np.asarray(first_x).reshape(-1)[0])
    y0 = float(np.asarray(first_y).reshape(-1)[0])

    i_negative = bool(int(projection.get("GRIB_iScansNegatively", 0)))
    j_positive = bool(int(projection.get("GRIB_jScansPositively", 1)))
    x_step = -float(dx) if i_negative else float(dx)
    y_step = float(dy) if j_positive else -float(dy)

    x_axis = x0 + x_step * np.arange(nx, dtype=np.float64)
    y_axis = y0 + y_step * np.arange(ny, dtype=np.float64)
    xx, yy = np.meshgrid(x_axis, y_axis)

    # Invert the spherical LCC so the profile retains useful geographic
    # coordinates for diagnostics/debugging. Sampling itself uses xx/yy.
    lat_0 = float(projection["GRIB_LaDInDegrees"])
    lon_0 = float(projection["GRIB_LoVInDegrees"])
    lat_1 = float(projection["GRIB_Latin1InDegrees"])
    lat_2 = float(projection["GRIB_Latin2InDegrees"])
    radius = 6371229.0
    phi0 = np.deg2rad(lat_0)
    phi1 = np.deg2rad(lat_1)
    phi2 = np.deg2rad(lat_2)
    lam0 = np.deg2rad(lon_0)

    def _t(phi):
        return np.tan(np.pi / 4.0 + phi / 2.0)

    if abs(lat_1 - lat_2) < 1.0e-8:
        n = np.sin(phi1)
    else:
        n = np.log(np.cos(phi1) / np.cos(phi2)) / np.log(_t(phi2) / _t(phi1))
    f = np.cos(phi1) * (_t(phi1) ** n) / n
    rho0 = radius * f / (_t(phi0) ** n)
    rho = np.sign(n) * np.sqrt(xx * xx + (rho0 - yy) * (rho0 - yy))
    rho = np.maximum(rho, 1.0e-6)
    theta = np.arctan2(xx, rho0 - yy)
    t = np.power(radius * f / rho, 1.0 / n)
    lat = 2.0 * np.arctan(t) - np.pi / 2.0
    lon = lam0 + theta / n
    return np.rad2deg(lat), np.rad2deg(lon), xx, yy

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
    # perform a projection for every MRMS chunk. Prefer the latitude/longitude
    # coordinates actually returned by pygrib when they have the decoded field
    # shape. Those coordinates are authoritative for a filtered NOMADS subset
    # and avoid subtle scan-direction/first-grid-point offsets. Only rebuild
    # the Lambert grid when pygrib returned malformed/broadcast coordinates.
    rap_shape = temp_c.shape[1:]
    raw_lat = np.asarray(data["latitude"], dtype=np.float64)
    raw_lon = np.asarray(data["longitude"], dtype=np.float64)
    if raw_lat.shape == rap_shape and raw_lon.shape == rap_shape:
        hx, hy = _lambert_conformal_conic_xy(
            raw_lon,
            raw_lat,
            data["projection"],
        )
    else:
        _, _, hx, hy = _build_regular_lambert_grid(
            rap_shape,
            data["projection"],
        )
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
    idx = np.searchsorted(sorted_values, values)
    idx = np.clip(idx, 1, len(sorted_values) - 1)
    left = idx - 1
    right = idx
    choose_right = np.abs(values - sorted_values[right]) < np.abs(values - sorted_values[left])
    return np.where(choose_right, right, left).astype(np.int64)


def sample_profile_to_mrms(profile: dict, lats: np.ndarray, lons: np.ndarray) -> dict:
    """Nearest-neighbor sample RAP profile fields onto an MRMS chunk."""
    lat = np.asarray(lats, dtype=np.float64)
    lon = np.asarray(lons, dtype=np.float64)
    if lat.ndim == 1:
        lat2, lon2 = lat[:, None], lon[None, :]
    else:
        lat2, lon2 = lat, lon

    # Reuse cached RAP projection coordinates.
    hx = np.asarray(profile.get("_projected_x"), dtype=np.float64)
    hy = np.asarray(profile.get("_projected_y"), dtype=np.float64)
    x_axis = np.asarray(profile.get("_x_axis"), dtype=np.float64)
    y_axis = np.asarray(profile.get("_y_axis"), dtype=np.float64)

    x_rev = x_axis[0] > x_axis[-1]
    y_rev = y_axis[0] > y_axis[-1]
    x_sorted = x_axis[::-1] if x_rev else x_axis
    y_sorted = y_axis[::-1] if y_rev else y_axis

    # Project the MRMS target coordinates with the same local spherical LCC
    # equation used for the RAP grid. This avoids the intermittent pyproj
    # mixed-dimensional-array failure and keeps source/target coordinates in
    # exactly the same projection space.
    target_shape = lon2.shape
    x, y = _lambert_conformal_conic_xy(lon2, lat2, profile["projection"])
    x = np.asarray(x, dtype=np.float64).reshape(target_shape)
    y = np.asarray(y, dtype=np.float64).reshape(target_shape)
    ix_sorted = _nearest_index(x_sorted, x)
    iy_sorted = _nearest_index(y_sorted, y)
    ix = (len(x_axis) - 1 - ix_sorted) if x_rev else ix_sorted
    iy = (len(y_axis) - 1 - iy_sorted) if y_rev else iy_sorted

    # Require the selected grid point itself to be close to the target
    # projected coordinate. This guards against accepting a clipped/offset
    # nearest index when a filtered RAP subset does not actually cover an MRMS
    # target point. Half a RAP grid spacing is the natural nearest-neighbor
    # acceptance radius, with a small tolerance for floating-point projection
    # roundoff.
    selected_x = x_axis[ix]
    selected_y = y_axis[iy]
    dx = float(np.nanmedian(np.abs(np.diff(x_sorted)))) if len(x_sorted) > 1 else np.nan
    dy = float(np.nanmedian(np.abs(np.diff(y_sorted)))) if len(y_sorted) > 1 else np.nan
    x_tolerance = 0.51 * dx if np.isfinite(dx) and dx > 0 else np.inf
    y_tolerance = 0.51 * dy if np.isfinite(dy) and dy > 0 else np.inf

    valid = (
        np.isfinite(x) & np.isfinite(y)
        & (x >= min(x_axis.min(), x_axis.max())) & (x <= max(x_axis.min(), x_axis.max()))
        & (y >= min(y_axis.min(), y_axis.max())) & (y <= max(y_axis.min(), y_axis.max()))
        & (np.abs(x - selected_x) <= x_tolerance)
        & (np.abs(y - selected_y) <= y_tolerance)
    )

    out: dict[str, np.ndarray] = {}
    for key in ("wetbulb_c", "temperature_c", "rh_ice_pct", "height_m"):
        src = np.asarray(profile[key])
        sampled = src[:, iy, ix].astype(np.float32, copy=False)
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
