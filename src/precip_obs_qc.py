from __future__ import annotations

"""Surface precipitation-type observations used as a local MRMS sanity check.

The core MRMS/RAP classifier remains the primary gridded diagnosis. This module
only applies a nearby-point observational check from AWC METARs (ASOS/AWOS)
and mPING reports before the winter mask/composite are rendered.

The observation feeds are queried server-side from GitHub Actions because the
AWC API does not permit CORS and both feeds are lightweight enough for the
5-minute live product cadence.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import os
import re
from pathlib import Path

import numpy as np
import requests

from classifier import (
    CLEAR,
    FZRA,
    MIXED,
    RAIN,
    SLEET,
    SNOW,
    UNKNOWN,
    ClassificationResult,
    rain_intensity_dbz,
)
from config import (
    AWC_METAR_URL,
    METAR_INFLUENCE_RADIUS_KM,
    METAR_MAX_AGE_MINUTES,
    MPING_INFLUENCE_RADIUS_KM,
    MPING_MAX_AGE_MINUTES,
    MPING_REPORTS_URL,
    NEUS_BOUNDS,
    OBS_QC_GRID_STRIDE,
    PRECIP_OBS_QC_ENABLED,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"

USER_AGENT = "WinterRadar/1.0 (NWS MRMS observational precip-type QC)"

METAR_TOKEN_RE = re.compile(
    r"(?<![A-Z])(?:SH|TS|FZ|MI|BC|PR|DR|BL)?"
    r"(?:DZ|RA|SN|SG|IC|PL|GR|GS|UP)(?![A-Z])",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ObsReport:
    source: str
    timestamp_utc: datetime
    lat: float
    lon: float
    phase: int
    description: str
    station: str = ""
    raw: str = ""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_time(value: object) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _in_neus(lat: float, lon: float) -> bool:
    south, west, north, east = NEUS_BOUNDS
    return south <= lat <= north and west <= lon <= east


def _metar_tokens(wx_string: str, raw_ob: str) -> list[str]:
    tokens = METAR_TOKEN_RE.findall(wx_string or "")
    if tokens:
        return [t.upper() for t in tokens]

    # Fallback for feeds/stations where wxString is empty. Pull the standard
    # present-weather groups from the raw METAR body without trying to decode
    # cloud, temperature, wind, or pressure groups.
    return [t.upper() for t in METAR_TOKEN_RE.findall(raw_ob or "")]


def _phase_from_metar(wx_string: str, raw_ob: str) -> tuple[int | None, str]:
    tokens = _metar_tokens(wx_string, raw_ob)
    if not tokens:
        return None, ""

    joined = " ".join(tokens)

    if "FZRA" in joined or "FZDZ" in joined:
        return FZRA, "Freezing precipitation"
    if "PL" in tokens and ("SN" in tokens or "SG" in tokens or "RA" in tokens):
        return MIXED, "Mixed precipitation"
    if "SN" in tokens or "SG" in tokens:
        if "RA" in tokens:
            return MIXED, "Rain/snow"
        return SNOW, "Snow"
    if "PL" in tokens:
        return SLEET, "Sleet/ice pellets"
    if "RA" in tokens or "DZ" in tokens:
        return RAIN, "Rain/drizzle"

    # Hail is not a winter-phase surface class for this gate. Unknown/lightly
    # coded weather is intentionally ignored rather than forcing a phase.
    return None, joined


def _extract_json_records(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        if isinstance(payload.get("results"), list):
            return [item for item in payload["results"] if isinstance(item, dict)]
        if isinstance(payload.get("data"), list):
            return [item for item in payload["data"] if isinstance(item, dict)]
    return []


def _request_json(
    session: requests.Session,
    url: str,
    params: dict,
    headers: dict | None = None,
) -> object:
    response = session.get(url, params=params, headers=headers, timeout=(10, 25))
    if response.status_code == 204:
        return []
    response.raise_for_status()
    return response.json()


def fetch_metar_reports(
    center_time: datetime | None = None,
    session: requests.Session | None = None,
) -> list[ObsReport]:
    center_time = center_time or _utc_now()
    session = session or requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    south, west, north, east = NEUS_BOUNDS
    bbox = f"{west},{south},{east},{north}"

    payload = _request_json(
        session,
        AWC_METAR_URL,
        params={
            "bbox": bbox,
            "format": "json",
            "hoursBeforeNow": 2,
        },
    )

    now = center_time
    reports: list[ObsReport] = []

    for item in _extract_json_records(payload):
        station = str(item.get("icaoId") or item.get("station") or "").strip().upper()
        # Limit the gate to U.S. ASOS/AWOS-style ICAO identifiers in the NEUS.
        # This avoids accidentally using nearby Canadian observations when the
        # operational intent is U.S. airport surface observations.
        if not station.startswith("K"):
            continue

        lat_raw = item.get("lat")
        lon_raw = item.get("lon")
        obs_time = _parse_time(item.get("obsTime") or item.get("obTime") or item.get("obtime"))
        if lat_raw is None or lon_raw is None or obs_time is None:
            continue

        try:
            lat = float(lat_raw)
            lon = float(lon_raw)
        except (TypeError, ValueError):
            continue

        if not _in_neus(lat, lon):
            continue

        age_minutes = (now - obs_time).total_seconds() / 60.0
        if age_minutes < -10.0 or age_minutes > METAR_MAX_AGE_MINUTES:
            continue

        wx = str(item.get("wxString") or "")
        raw = str(item.get("rawOb") or "")
        phase, description = _phase_from_metar(wx, raw)
        if phase is None:
            continue

        reports.append(
            ObsReport(
                source="ASOS/AWOS",
                timestamp_utc=obs_time,
                lat=lat,
                lon=lon,
                phase=phase,
                description=description,
                station=station,
                raw=wx or raw,
            )
        )

    # One station can appear more than once in the recent response. Keep the
    # newest usable present-weather report at each airport.
    latest: dict[str, ObsReport] = {}
    for report in reports:
        existing = latest.get(report.station)
        if existing is None or report.timestamp_utc > existing.timestamp_utc:
            latest[report.station] = report
    return sorted(latest.values(), key=lambda r: r.timestamp_utc)


def _mping_feature_records(payload: object) -> list[dict]:
    if not isinstance(payload, dict):
        return []
    results = payload.get("results")
    if not isinstance(results, list):
        return []
    return [item for item in results if isinstance(item, dict)]


def _phase_from_mping(description: str) -> int | None:
    d = " ".join(str(description or "").strip().lower().split())
    if not d:
        return None
    if "freezing rain" in d or "freezing drizzle" in d:
        return FZRA
    if "mixed rain and snow" in d:
        return MIXED
    if "mixed ice pellets and snow" in d:
        return MIXED
    if "mixed rain and ice pellets" in d:
        return MIXED
    if "sleet" in d or "ice pellets" in d:
        return SLEET
    if d == "snow" or " snow" in d:
        return SNOW
    if d == "rain" or " rain" in d or d == "drizzle":
        return RAIN
    return None


def fetch_mping_reports(
    center_time: datetime | None = None,
    session: requests.Session | None = None,
) -> list[ObsReport]:
    center_time = center_time or _utc_now()
    session = session or requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})

    south, west, north, east = NEUS_BOUNDS
    lower = center_time - timedelta(minutes=MPING_MAX_AGE_MINUTES)

    params = {
        "category": "Rain/Snow",
        "in_bbox": f"{west},{south},{east},{north}",
        "obtime_gte": lower.strftime("%Y-%m-%d %H:%M:%S"),
    }

    api_key = (
        os.environ.get("MPING_API_KEY")
        or os.environ.get("MPING_API_TOKEN")
        or os.environ.get("MPING_TOKEN")
    )
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Token {api_key}"

    payload = _request_json(session, MPING_REPORTS_URL, params=params, headers=headers)
    reports: list[ObsReport] = []

    for item in _mping_feature_records(payload):
        geom = item.get("geom") or item.get("geometry") or {}
        coords = geom.get("coordinates") if isinstance(geom, dict) else None
        if not isinstance(coords, (list, tuple)) or len(coords) < 2:
            continue

        try:
            lon = float(coords[0])
            lat = float(coords[1])
        except (TypeError, ValueError):
            continue

        if not _in_neus(lat, lon):
            continue

        obs_time = _parse_time(item.get("obtime") or item.get("obsTime"))
        if obs_time is None:
            continue

        age_minutes = (center_time - obs_time).total_seconds() / 60.0
        if age_minutes < -10.0 or age_minutes > MPING_MAX_AGE_MINUTES:
            continue

        description = str(item.get("description") or "").strip()
        phase = _phase_from_mping(description)
        if phase is None:
            continue

        reports.append(
            ObsReport(
                source="mPING",
                timestamp_utc=obs_time,
                lat=lat,
                lon=lon,
                phase=phase,
                description=description,
                station=f"mPING:{item.get('id', '')}",
                raw=description,
            )
        )

    return reports


def _coarse_observation_field(
    reports: list[ObsReport],
    lats: np.ndarray,
    lons: np.ndarray,
    center_time: datetime,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Rasterize nearest observations onto a coarse NEUS grid.

    Only cells inside the NEUS bounding box are evaluated. The nearest usable
    point observation wins; the source-specific influence radius prevents a
    distant airport report from changing a broad gridded precipitation field.
    """

    lat_arr = np.asarray(lats, dtype=np.float64)
    lon_arr = np.asarray(lons, dtype=np.float64)
    if lat_arr.ndim != 1 or lon_arr.ndim != 1:
        if lat_arr.ndim == 2 and lon_arr.ndim == 2:
            lat_arr = lat_arr[:, 0]
            lon_arr = lon_arr[0, :]
        else:
            raise ValueError("Observation QC requires 1-D or matching 2-D MRMS coordinates.")

    south, west, north, east = NEUS_BOUNDS
    y_mask = (lat_arr >= south) & (lat_arr <= north)
    x_mask = (lon_arr >= west) & (lon_arr <= east)
    if not np.any(y_mask) or not np.any(x_mask):
        return (
            np.full((1, 1), -1, dtype=np.int8),
            np.full((1, 1), np.inf, dtype=np.float32),
            {"regional_cells": 0},
        )

    y_idx = np.flatnonzero(y_mask)
    x_idx = np.flatnonzero(x_mask)
    stride = max(1, int(OBS_QC_GRID_STRIDE))
    y_idx = y_idx[::stride]
    x_idx = x_idx[::stride]

    grid_lat = lat_arr[y_idx]
    grid_lon = lon_arr[x_idx]
    yy, xx = np.meshgrid(grid_lat, grid_lon, indexing="ij")

    best_phase = np.full(yy.shape, -1, dtype=np.int8)
    best_distance = np.full(yy.shape, np.inf, dtype=np.float32)
    best_report = np.full(yy.shape, -1, dtype=np.int16)

    report_meta: list[ObsReport] = []

    for idx, report in enumerate(reports):
        radius_km = (
            METAR_INFLUENCE_RADIUS_KM
            if report.source == "ASOS/AWOS"
            else MPING_INFLUENCE_RADIUS_KM
        )
        age_minutes = max(
            0.0,
            (center_time - report.timestamp_utc).total_seconds() / 60.0,
        )
        if report.source == "ASOS/AWOS" and age_minutes > METAR_MAX_AGE_MINUTES:
            continue
        if report.source == "mPING" and age_minutes > MPING_MAX_AGE_MINUTES:
            continue

        mean_lat = np.deg2rad((yy + report.lat) * 0.5)
        dlat_km = (yy - report.lat) * 111.2
        dlon_km = (xx - report.lon) * 111.2 * np.cos(mean_lat)
        distance = np.hypot(dlat_km, dlon_km).astype(np.float32)

        candidate = distance <= radius_km
        candidate &= distance < best_distance
        if not np.any(candidate):
            continue

        best_distance[candidate] = distance[candidate]
        best_phase[candidate] = np.int8(report.phase)
        best_report[candidate] = np.int16(idx)
        if idx >= len(report_meta):
            report_meta.append(report)

    # Repeating the coarse field back onto the native regional subset is enough
    # for a point-observation gate; it avoids building a multi-million-cell
    # distance field for the whole CONUS.
    meta = {
        "regional_y_indices": y_idx.tolist(),
        "regional_x_indices": x_idx.tolist(),
        "regional_cells": int(yy.size),
        "reports": report_meta,
    }
    return best_phase, best_distance, meta


def apply_observation_qc(
    result: ClassificationResult,
    reflectivity: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    mrms_time_utc: str,
) -> tuple[ClassificationResult, dict]:
    """Apply the nearby ASOS/AWOS + mPING surface-phase sanity check."""

    if not PRECIP_OBS_QC_ENABLED:
        return result, {"enabled": False, "status": "disabled"}

    base_time = _parse_time(mrms_time_utc) or _utc_now()
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    diagnostics = {
        "enabled": True,
        "status": "ok",
        "mrms_time_utc": base_time.isoformat(),
        "domain": {
            "name": "Northeast U.S.",
            "bounds": list(NEUS_BOUNDS),
        },
        "rules": {
            "metar_max_age_minutes": METAR_MAX_AGE_MINUTES,
            "metar_influence_radius_km": METAR_INFLUENCE_RADIUS_KM,
            "mping_max_age_minutes": MPING_MAX_AGE_MINUTES,
            "mping_influence_radius_km": MPING_INFLUENCE_RADIUS_KM,
            "grid_stride": OBS_QC_GRID_STRIDE,
            "rain_report": "nearby observed rain forces local classified phase to rain",
            "winter_report": "nearby observed snow/sleet/freezing-rain/mixed can replace rain/unknown and validate winter types",
            "no_precip_report": "not used as negative evidence",
        },
    }

    try:
        metar_reports = fetch_metar_reports(base_time, session=session)
    except Exception as exc:
        metar_reports = []
        diagnostics["metar_error"] = f"{type(exc).__name__}: {exc}"
    try:
        mping_reports = fetch_mping_reports(base_time, session=session)
    except Exception as exc:
        mping_reports = []
        diagnostics["mping_error"] = f"{type(exc).__name__}: {exc}"

    reports = metar_reports + mping_reports
    diagnostics["metar_reports"] = len(metar_reports)
    diagnostics["mping_reports"] = len(mping_reports)

    if not reports:
        diagnostics["status"] = "no_usable_reports"
        diagnostics["reports_used"] = 0
        return result, diagnostics

    obs_phase, obs_distance, raster_meta = _coarse_observation_field(
        reports,
        lats,
        lons,
        base_time,
    )

    lat_arr = np.asarray(lats, dtype=np.float64)
    lon_arr = np.asarray(lons, dtype=np.float64)
    if lat_arr.ndim == 2:
        lat_arr = lat_arr[:, 0]
    if lon_arr.ndim == 2:
        lon_arr = lon_arr[0, :]

    south, west, north, east = NEUS_BOUNDS
    y_full = np.flatnonzero((lat_arr >= south) & (lat_arr <= north))
    x_full = np.flatnonzero((lon_arr >= west) & (lon_arr <= east))

    if y_full.size == 0 or x_full.size == 0:
        diagnostics["status"] = "domain_not_on_grid"
        return result, diagnostics

    # Rebuild the coarse grid at the actual native regional indices. The coarse
    # calculation is expanded into the same stride blocks used to evaluate it.
    y0, x0 = y_full[0], x_full[0]
    y1, x1 = y_full[-1] + 1, x_full[-1] + 1
    expanded = np.full((y1 - y0, x1 - x0), -1, dtype=np.int8)
    expanded_dist = np.full((y1 - y0, x1 - x0), np.inf, dtype=np.float32)

    coarse_y = np.asarray(raster_meta["regional_y_indices"], dtype=np.int64)
    coarse_x = np.asarray(raster_meta["regional_x_indices"], dtype=np.int64)
    local_y = coarse_y - y0
    local_x = coarse_x - x0

    for j, gy in enumerate(local_y):
        if gy < 0 or gy >= expanded.shape[0]:
            continue
        for i, gx in enumerate(local_x):
            if gx < 0 or gx >= expanded.shape[1]:
                continue
            expanded[gy, gx] = obs_phase[j, i]
            expanded_dist[gy, gx] = obs_distance[j, i]

    # Fill each stride block with its coarse representative. Edge blocks are
    # trimmed so the array stays aligned with the native regional MRMS crop.
    if expanded.size:
        for yy0 in range(0, expanded.shape[0], max(1, OBS_QC_GRID_STRIDE)):
            yy1 = min(yy0 + OBS_QC_GRID_STRIDE, expanded.shape[0])
            gy = min(yy0 // max(1, OBS_QC_GRID_STRIDE), obs_phase.shape[0] - 1)
            for xx0 in range(0, expanded.shape[1], max(1, OBS_QC_GRID_STRIDE)):
                xx1 = min(xx0 + OBS_QC_GRID_STRIDE, expanded.shape[1])
                gx = min(xx0 // max(1, OBS_QC_GRID_STRIDE), obs_phase.shape[1] - 1)
                expanded[yy0:yy1, xx0:xx1] = obs_phase[gy, gx]
                expanded_dist[yy0:yy1, xx0:xx1] = obs_distance[gy, gx]

    obs_full = np.full(result.phase.shape, -1, dtype=np.int8)
    dist_full = np.full(result.phase.shape, np.inf, dtype=np.float32)
    obs_full[y0:y1, x0:x1] = expanded
    dist_full[y0:y1, x0:x1] = expanded_dist

    ref = np.asarray(reflectivity, dtype=np.float32)
    phase = np.array(result.phase, copy=True)
    confidence = np.array(result.confidence, copy=True)
    intensity = np.array(result.intensity, copy=True)

    precip = np.isfinite(ref) & (ref >= 10.0)
    touched = precip & (obs_full >= 0) & np.isfinite(dist_full)

    report_counts = {
        "rain": 0,
        "snow": 0,
        "sleet": 0,
        "freezing_rain": 0,
        "mixed": 0,
    }
    adjustments = {
        "rain_forced": 0,
        "winter_promoted": 0,
        "winter_confirmed": 0,
    }

    for code, name in (
        (RAIN, "rain"),
        (SNOW, "snow"),
        (SLEET, "sleet"),
        (FZRA, "freezing_rain"),
        (MIXED, "mixed"),
    ):
        cells = touched & (obs_full == code)
        report_counts[name] = int(np.count_nonzero(cells))

        if not np.any(cells):
            continue

        if code == RAIN:
            changed = cells & (phase != RAIN)
            phase[changed] = RAIN
            confidence[changed] = np.maximum(confidence[changed], 0.72)
            intensity[changed] = rain_intensity_dbz(ref[changed])
            adjustments["rain_forced"] += int(np.count_nonzero(changed))
        else:
            changed = cells & ((phase == RAIN) | (phase == UNKNOWN) | (phase == CLEAR))
            phase[changed] = code
            confidence[changed] = np.maximum(confidence[changed], 0.66)
            intensity[changed] = 0
            adjustments["winter_promoted"] += int(np.count_nonzero(changed))

            confirmed = cells & (phase == code) & ~changed
            confidence[confirmed] = np.maximum(confidence[confirmed], 0.72)
            adjustments["winter_confirmed"] += int(np.count_nonzero(confirmed))

    # Never invent precipitation where MRMS does not show precipitation.
    phase[~precip] = CLEAR
    confidence[~precip] = 0.0
    intensity[~precip] = 0

    diagnostics["report_cell_counts"] = report_counts
    diagnostics["adjustments"] = adjustments
    diagnostics["reports_used"] = int(np.count_nonzero(touched))
    diagnostics["cells_affected"] = int(np.count_nonzero(touched))
    diagnostics["distance_max_km"] = (
        float(np.nanmax(dist_full[touched])) if np.any(touched) else None
    )
    diagnostics["status"] = (
        "applied" if np.any(touched) else "reports_outside_radar_precip"
    )
    diagnostics["stations"] = [
        {
            "source": report.source,
            "station": report.station,
            "timestamp_utc": report.timestamp_utc.isoformat(),
            "lat": report.lat,
            "lon": report.lon,
            "phase": report.phase,
            "description": report.description,
        }
        for report in reports
    ]

    updated = ClassificationResult(
        phase=phase,
        confidence=confidence,
        intensity=intensity,
    )
    return updated, diagnostics


def write_diagnostics(diagnostics: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "precip_obs_qc.json").write_text(
        json.dumps(diagnostics, indent=2),
        encoding="utf-8",
    )


def self_test() -> None:
    from datetime import datetime

    assert _phase_from_metar("-SN", "")[0] == SNOW
    assert _phase_from_metar("FZRA", "")[0] == FZRA
    assert _phase_from_metar("RASN", "")[0] == MIXED
    assert _phase_from_metar("PL", "")[0] == SLEET
    assert _phase_from_metar("-RA", "")[0] == RAIN
    assert _phase_from_mping("Freezing Rain") == FZRA
    assert _phase_from_mping("Sleet/Ice Pellets") == SLEET
    assert _phase_from_mping("Mixed Rain and Snow") == MIXED

    ref = np.full((100, 100), 25.0, dtype=np.float32)
    result = ClassificationResult(
        phase=np.full((100, 100), SNOW, dtype=np.uint8),
        confidence=np.full((100, 100), 0.8, dtype=np.float32),
        intensity=np.zeros((100, 100), dtype=np.uint8),
    )
    lats = np.linspace(48.5, 37.0, 100)
    lons = np.linspace(-84.5, -66.0, 100)

    report = ObsReport(
        source="ASOS/AWOS",
        timestamp_utc=datetime(2026, 9, 28, 17, 0, tzinfo=timezone.utc),
        lat=44.0,
        lon=-73.0,
        phase=RAIN,
        description="Rain",
        station="KBTV",
    )
    center = report.timestamp_utc
    coarse, distance, meta = _coarse_observation_field([report], lats, lons, center)
    assert coarse.shape[0] > 0 and coarse.shape[1] > 0
    assert np.any(coarse == RAIN)
    assert np.any(np.isfinite(distance))

    print("PRECIP OBSERVATION QC SELF-TEST PASSED")


if __name__ == "__main__":
    self_test()
