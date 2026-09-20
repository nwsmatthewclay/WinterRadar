from __future__ import annotations

"""Archive timestamped MRMS observations and phase masks for WinterRadar.

The live WinterRadar pipeline intentionally continues to use the `latest`
MRMS product. This module is a secondary history collector. Each time the
GitHub Actions workflow runs it:

1. Reads NOAA's timestamped ReflectivityAtLowestAltitude directory.
2. Finds all observations from the previous 24 hours.
3. Looks at the two daily GitHub Releases used by WinterRadar as persistent
   storage and determines which observations are missing.
4. Downloads and renders only those missing observations.
5. Stores compact WebP frames in the appropriate daily release.
6. Archives one winter-phase mask per 10-minute bucket, keeping the release
   comfortably below GitHub's 1,000-asset-per-release limit.
7. Builds outputs/mrms_history.json for the Pages viewer.

The live radar path is never modified by this script. The workflow should run
this step with `continue-on-error: true` so a history/API problem cannot take
the current radar offline.
"""

import argparse
import gzip
import io
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from html import unescape
from pathlib import Path
from typing import Iterable
from urllib.parse import quote

import numpy as np
import requests
from PIL import Image

from render import reflectivity_to_rgba

def _load_eccodes():
    try:
        from eccodes import (
            codes_get_values,
            codes_grib_new_from_file,
            codes_release,
        )
        return codes_get_values, codes_grib_new_from_file, codes_release
    except ImportError as exc:  # pragma: no cover - exercised on Actions, not local container
        raise RuntimeError(
            "Python eccodes is required for MRMS history decoding. "
            "It is installed as a dependency of cfgrib in the GitHub Actions workflow."
        ) from exc


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
MRMS_HISTORY_FILE = OUTPUT_DIR / "mrms_history.json"

MRMS_REFLECTIVITY_DIR = (
    "https://mrms.ncep.noaa.gov/2D/ReflectivityAtLowestAltitude/"
)

MRMS_FILENAME_RE = re.compile(
    r"MRMS_ReflectivityAtLowestAltitude_00\.50_(\d{8}-\d{6})\.grib2\.gz"
)

RADAR_ASSET_RE = re.compile(r"^radar_(\d{8}-\d{6})\.webp$")
PHASE_ASSET_RE = re.compile(r"^phase_(\d{8}-\d{4})\.webp$")

# The existing viewer's regional map area. History is intentionally stored at
# this resolution/extent rather than carrying the full CONUS raster for every
# frame. It fully contains the existing BTV CWA bounds and the current
# Regional view.
HISTORY_FALLBACK_BOUNDS = [
    [40.95, -77.50],
    [46.15, -68.40],
]

# Archive all observed ~2-minute MRMS frames. Phase masks are bucketed to one
# per 10 minutes to leave room under GitHub's 1,000-assets-per-release limit.
HISTORY_HOURS = 24
PHASE_BUCKET_MINUTES = 10
MAX_NEW_RADAR_FRAMES_PER_RUN = int(
    os.environ.get("MRMS_HISTORY_MAX_FRAMES_PER_RUN", "40")
)

REQUEST_TIMEOUT = (20, 120)
UPLOAD_TIMEOUT = (20, 180)
USER_AGENT = "WinterRadar/1.1 (MRMS 24-hour history collector)"
GITHUB_API_VERSION = "2026-03-10"
HISTORY_DEBUG = os.environ.get("MRMS_HISTORY_DEBUG", "0") == "1"


@dataclass(frozen=True)
class Observation:
    valid_time: datetime
    filename: str
    source_url: str

    @property
    def timestamp_key(self) -> str:
        return self.valid_time.strftime("%Y%m%d-%H%M%S")

    @property
    def asset_name(self) -> str:
        return f"radar_{self.timestamp_key}.webp"


@dataclass(frozen=True)
class ReleaseInfo:
    tag: str
    release_id: int
    upload_url: str
    html_url: str
    assets: dict[str, dict]


class GitHubReleaseStore:
    def __init__(self, token: str, repository: str) -> None:
        if not token:
            raise RuntimeError("GITHUB_TOKEN is not set.")
        if not repository or "/" not in repository:
            raise RuntimeError(f"Invalid GITHUB_REPOSITORY: {repository!r}")
        self.token = token
        self.repository = repository
        self.api_root = "https://api.github.com"
        self.headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
            "User-Agent": USER_AGENT,
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def _url(self, path: str) -> str:
        return f"{self.api_root}/repos/{self.repository}{path}"

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        response = self.session.request(
            method,
            self._url(path),
            timeout=REQUEST_TIMEOUT,
            **kwargs,
        )
        if not response.ok:
            detail = response.text[:500].replace("\n", " ")
            raise RuntimeError(
                f"GitHub API {method} {path} failed: HTTP {response.status_code}: {detail}"
            )
        return response

    def get_release(self, tag: str) -> ReleaseInfo | None:
        response = self.session.get(
            self._url(f"/releases/tags/{quote(tag, safe='')}"),
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code == 404:
            return None
        if not response.ok:
            detail = response.text[:500].replace("\n", " ")
            raise RuntimeError(
                f"GitHub release lookup failed for {tag}: "
                f"HTTP {response.status_code}: {detail}"
            )
        data = response.json()
        assets = self.list_assets(int(data["id"]))
        upload_url = str(data["upload_url"]).split("{", 1)[0]
        return ReleaseInfo(
            tag=str(data["tag_name"]),
            release_id=int(data["id"]),
            upload_url=upload_url,
            html_url=str(data["html_url"]),
            assets=assets,
        )

    def create_release(self, tag: str) -> ReleaseInfo:
        payload = {
            "tag_name": tag,
            "name": f"WinterRadar MRMS History — {tag.removeprefix('mrms-')}",
            "body": (
                "Automated 24-hour WinterRadar MRMS observation archive. "
                "Radar frames are timestamped observations; phase masks are "
                "10-minute snapshots used by the time-history viewer."
            ),
            "draft": False,
            "prerelease": False,
            "generate_release_notes": False,
        }
        response = self._request("POST", "/releases", json=payload)
        data = response.json()
        print(f"  Created GitHub release {tag}: {data['html_url']}")
        return ReleaseInfo(
            tag=str(data["tag_name"]),
            release_id=int(data["id"]),
            upload_url=str(data["upload_url"]).split("{", 1)[0],
            html_url=str(data["html_url"]),
            assets={},
        )

    def ensure_release(self, tag: str) -> ReleaseInfo:
        release = self.get_release(tag)
        if release is not None:
            return release
        return self.create_release(tag)

    def list_assets(self, release_id: int) -> dict[str, dict]:
        assets: dict[str, dict] = {}
        page = 1
        while True:
            response = self.session.get(
                self._url(f"/releases/{release_id}/assets"),
                params={"per_page": 100, "page": page},
                timeout=REQUEST_TIMEOUT,
            )
            if not response.ok:
                detail = response.text[:500].replace("\n", " ")
                raise RuntimeError(
                    f"GitHub asset listing failed for release {release_id}: "
                    f"HTTP {response.status_code}: {detail}"
                )
            batch = response.json()
            for asset in batch:
                assets[str(asset["name"])] = asset
            if len(batch) < 100:
                break
            page += 1
        return assets

    def upload_asset(
        self,
        release: ReleaseInfo,
        asset_name: str,
        data: bytes,
        content_type: str = "image/webp",
    ) -> dict:
        params = {"name": asset_name}
        headers = dict(self.headers)
        headers["Content-Type"] = content_type
        response = self.session.post(
            release.upload_url,
            params=params,
            headers=headers,
            data=data,
            timeout=UPLOAD_TIMEOUT,
        )
        if response.status_code == 422:
            # A previous attempt may have succeeded while the client timed out.
            # Re-list assets and treat an existing matching name as success.
            refreshed = self.list_assets(release.release_id)
            if asset_name in refreshed:
                return refreshed[asset_name]
        if not response.ok:
            detail = response.text[:500].replace("\n", " ")
            raise RuntimeError(
                f"GitHub asset upload failed for {asset_name}: "
                f"HTTP {response.status_code}: {detail}"
            )
        return response.json()

    def delete_release(self, release_id: int) -> None:
        self._request("DELETE", f"/releases/{release_id}")

    def delete_tag_ref(self, tag: str) -> None:
        response = self.session.delete(
            self._url(f"/git/refs/tags/{quote(tag, safe='')}"),
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code not in (204, 404):
            detail = response.text[:500].replace("\n", " ")
            raise RuntimeError(
                f"Git tag deletion failed for {tag}: HTTP {response.status_code}: {detail}"
            )

    def list_history_releases(self) -> list[dict]:
        response = self.session.get(
            self._url("/releases"),
            params={"per_page": 100},
            timeout=REQUEST_TIMEOUT,
        )
        if not response.ok:
            detail = response.text[:500].replace("\n", " ")
            raise RuntimeError(
                f"GitHub release listing failed: HTTP {response.status_code}: {detail}"
            )
        return [
            release
            for release in response.json()
            if str(release.get("tag_name", "")).startswith("mrms-")
        ]


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def parse_observation_timestamp(value: str) -> datetime:
    return datetime.strptime(value, "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc)


def floor_time(value: datetime, minutes: int) -> datetime:
    value = value.astimezone(timezone.utc).replace(second=0, microsecond=0)
    return value - timedelta(minutes=value.minute % minutes)


def parse_mrms_directory(html: str) -> list[Observation]:
    observations: dict[datetime, Observation] = {}
    for match in MRMS_FILENAME_RE.finditer(unescape(html)):
        stamp = match.group(1)
        valid_time = parse_observation_timestamp(stamp)
        filename = match.group(0)
        observations[valid_time] = Observation(
            valid_time=valid_time,
            filename=filename,
            source_url=MRMS_REFLECTIVITY_DIR + quote(filename, safe="_-.") ,
        )
    return sorted(observations.values(), key=lambda item: item.valid_time)


def fetch_mrms_directory(session: requests.Session) -> list[Observation]:
    response = session.get(
        MRMS_REFLECTIVITY_DIR,
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    observations = parse_mrms_directory(response.text)
    if not observations:
        raise RuntimeError(
            "No timestamped ReflectivityAtLowestAltitude observations were found "
            "in the MRMS directory."
        )
    return observations


def read_history_bounds(metadata_path: Path) -> tuple[float, float, float, float]:
    """Return (south, west, north, east) for the regional archive crop."""
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            live_bounds = metadata.get("bounds")
            if isinstance(live_bounds, list) and len(live_bounds) == 4:
                # The live full MRMS bounds are intentionally ignored as the
                # archive uses a smaller regional crop.
                pass
        except Exception:
            pass

    south = HISTORY_FALLBACK_BOUNDS[0][0]
    west = HISTORY_FALLBACK_BOUNDS[0][1]
    north = HISTORY_FALLBACK_BOUNDS[1][0]
    east = HISTORY_FALLBACK_BOUNDS[1][1]
    return south, west, north, east


def load_source_coordinates() -> tuple[np.ndarray, np.ndarray]:
    lat_path = OUTPUT_DIR / "latitude.npy"
    lon_path = OUTPUT_DIR / "longitude.npy"
    if lat_path.exists() and lon_path.exists():
        lats = np.asarray(np.load(lat_path), dtype=np.float64)
        lons = np.asarray(np.load(lon_path), dtype=np.float64)
        if lats.ndim == 1 and lons.ndim == 1:
            return lats, lons

    # Native MRMS grid used by WinterRadar. This is only a fallback so the
    # history collector can still operate if a future live-core refactor stops
    # saving latitude.npy/longitude.npy.
    lats = np.linspace(54.995, 20.005, 3500, dtype=np.float64)
    lons = np.linspace(-129.995, -60.005, 7000, dtype=np.float64)
    return lats, lons


def crop_indices(
    lats: np.ndarray,
    lons: np.ndarray,
    bounds: tuple[float, float, float, float],
) -> tuple[int, int, int, int, tuple[float, float, float, float]]:
    south, west, north, east = bounds
    lat_mask = (lats >= south) & (lats <= north)
    lon_mask = (lons >= west) & (lons <= east)
    if not np.any(lat_mask) or not np.any(lon_mask):
        raise RuntimeError(f"History crop does not overlap MRMS grid: {bounds}")

    y_indices = np.flatnonzero(lat_mask)
    x_indices = np.flatnonzero(lon_mask)
    y0, y1 = int(y_indices.min()), int(y_indices.max()) + 1
    x0, x1 = int(x_indices.min()), int(x_indices.max()) + 1

    actual_south = float(min(lats[y0], lats[y1 - 1]))
    actual_north = float(max(lats[y0], lats[y1 - 1]))
    actual_west = float(min(lons[x0], lons[x1 - 1]))
    actual_east = float(max(lons[x0], lons[x1 - 1]))
    return y0, y1, x0, x1, (actual_south, actual_west, actual_north, actual_east)


def mercator_y(lat_deg: np.ndarray | float) -> np.ndarray | float:
    max_lat = 85.0511287798
    lat = np.clip(np.asarray(lat_deg, dtype=np.float64), -max_lat, max_lat)
    phi = np.deg2rad(lat)
    return np.log(np.tan(np.pi / 4.0 + phi / 2.0))


def inverse_mercator_lat(y: np.ndarray) -> np.ndarray:
    return np.rad2deg(2.0 * np.arctan(np.exp(y)) - np.pi / 2.0)


def build_webmercator_row_map(
    source_height: int,
    south: float,
    north: float,
) -> tuple[int, np.ndarray]:
    y_s = float(mercator_y(south))
    y_n = float(mercator_y(north))
    mercator_span = y_n - y_s
    geographic_span_rad = math.radians(north - south)
    output_height = max(
        2,
        int(round(source_height * mercator_span / geographic_span_rad)),
    )

    y_centers = y_n - (
        np.arange(output_height, dtype=np.float64) + 0.5
    ) * mercator_span / output_height
    lat_centers = inverse_mercator_lat(y_centers)
    source_rows = (north - lat_centers) / (north - south) * (source_height - 1)
    nearest_rows = np.rint(np.clip(source_rows, 0.0, source_height - 1)).astype(np.int32)
    return output_height, nearest_rows


def subset_to_webmercator(
    rgba: np.ndarray,
    crop_bounds: tuple[float, float, float, float],
    row_map_cache: dict[tuple[int, float, float], tuple[int, np.ndarray]],
) -> np.ndarray:
    south, _west, north, _east = crop_bounds
    cache_key = (int(rgba.shape[0]), round(south, 6), round(north, 6))
    if cache_key not in row_map_cache:
        row_map_cache[cache_key] = build_webmercator_row_map(
            rgba.shape[0], south, north
        )
    _output_height, nearest_rows = row_map_cache[cache_key]
    return np.asarray(rgba[nearest_rows], dtype=np.uint8)


def rgba_to_webp_bytes(rgba: np.ndarray, quality: int = 88) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(
        buffer,
        format="WEBP",
        quality=quality,
        method=6,
    )
    return buffer.getvalue()


def phase_png_to_webp(
    path: Path,
    crop: tuple[int, int, int, int],
    crop_bounds: tuple[float, float, float, float],
    row_map_cache: dict[tuple[int, float, float], tuple[int, np.ndarray]],
) -> bytes:
    y0, y1, x0, x1 = crop
    with Image.open(path) as im:
        rgba = np.asarray(im.convert("RGBA"))[y0:y1, x0:x1]
    projected = subset_to_webmercator(rgba, crop_bounds, row_map_cache)
    return rgba_to_webp_bytes(projected, quality=90)


def decode_reflectivity_from_file(
    grib_path: Path,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    """Decode one timestamped MRMS GRIB using ecCodes from a real file handle.

    ecCodes' ``codes_grib_new_from_file`` expects a file object backed by a
    real OS file descriptor. Feeding it ``io.BytesIO`` or ``gzip.GzipFile``
    can produce the opaque ``fileno`` failure seen in GitHub Actions. The
    gzip member is therefore fully decompressed to disk first, then opened
    in binary mode for ecCodes.
    """
    # Keep this unpacking in lockstep with _load_eccodes().  The history
    # decoder only needs these three ecCodes calls.
    codes_get_values, codes_grib_new_from_file, codes_release = _load_eccodes()

    with grib_path.open("rb") as grib_file:
        while True:
            gid = codes_grib_new_from_file(grib_file)
            if gid is None:
                break
            try:
                values = codes_get_values(gid)
                arr = np.asarray(values, dtype=np.float32)
                if arr.size == expected_shape[0] * expected_shape[1] and arr.ndim == 1:
                    # This timestamped MRMS directory contains the requested
                    # reflectivity product, so the full expected grid size is
                    # the safest discriminator. Do not depend on shortName
                    # spelling, which has varied across ecCodes/MRMS files.
                    arr = arr.reshape(expected_shape)
                    arr = arr.copy()
                    arr[arr <= -900.0] = np.nan
                    return arr
            finally:
                codes_release(gid)

    raise RuntimeError(
        f"Could not decode a {expected_shape[0]}x{expected_shape[1]} MRMS reflectivity field "
        f"from {grib_path.name}."
    )


def download_and_render_observation(
    session: requests.Session,
    observation: Observation,
    expected_shape: tuple[int, int],
    crop: tuple[int, int, int, int],
    crop_bounds: tuple[float, float, float, float],
    row_map_cache: dict[tuple[int, float, float], tuple[int, np.ndarray]],
) -> bytes:
    """Download one MRMS frame, decode it from a real file, and return WebP bytes."""
    print(f"    Downloading {observation.filename}")

    # ecCodes requires a real file descriptor. Keep both the compressed MRMS
    # payload and the decompressed GRIB on disk only for the duration of one
    # frame, then remove them in the finally block. This also avoids holding
    # the full 3500x7000 GRIB payload in Python memory.
    gz_tmp = tempfile.NamedTemporaryFile(
        prefix="mrms_history_",
        suffix=".grib2.gz",
        delete=False,
    )
    gz_path = Path(gz_tmp.name)
    gz_tmp.close()

    grib_fd, grib_name = tempfile.mkstemp(
        prefix="mrms_history_",
        suffix=".grib2",
    )
    os.close(grib_fd)
    grib_path = Path(grib_name)

    try:
        with session.get(
            observation.source_url,
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": USER_AGENT, "Accept": "application/octet-stream"},
            stream=True,
        ) as response:
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "").lower()
            print(f"      HTTP {response.status_code}; content-type={content_type or 'unknown'}")

            bytes_written = 0
            with gz_path.open("wb") as out_file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    out_file.write(chunk)
                    bytes_written += len(chunk)

            if bytes_written < 32:
                raise RuntimeError(
                    f"MRMS historical response is unexpectedly small ({bytes_written} bytes): "
                    f"{observation.filename}"
                )

            with gz_path.open("rb") as check:
                magic = check.read(2)
            if magic != b"\x1f\x8b":
                raise RuntimeError(
                    f"MRMS historical response is not gzip data: {observation.filename}"
                )

        # Fully validate/decompress the gzip member to a physical GRIB file.
        with gzip.open(gz_path, "rb") as source, grib_path.open("wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)

        with grib_path.open("rb") as check:
            if check.read(4) != b"GRIB":
                raise RuntimeError(
                    f"Decompressed MRMS payload is not a GRIB file: {observation.filename}"
                )

        reflectivity = decode_reflectivity_from_file(grib_path, expected_shape)

        y0, y1, x0, x1 = crop
        regional = reflectivity[y0:y1, x0:x1]
        rgba = reflectivity_to_rgba(regional)
        del reflectivity, regional

        projected = subset_to_webmercator(rgba, crop_bounds, row_map_cache)
        del rgba

        data = rgba_to_webp_bytes(projected, quality=88)
        print(
            f"      Decoded/rendered {observation.asset_name}: "
            f"{len(data):,} WebP bytes"
        )
        return data

    finally:
        gz_path.unlink(missing_ok=True)
        grib_path.unlink(missing_ok=True)


def phase_bucket_for_timestamp(value: datetime) -> datetime:
    return floor_time(value, PHASE_BUCKET_MINUTES)


def existing_phase_bucket_names(assets: Iterable[str]) -> set[str]:
    buckets: set[str] = set()
    for name in assets:
        match = PHASE_ASSET_RE.match(name)
        if match:
            buckets.add(match.group(1))
    return buckets


def existing_radar_timestamps(assets: Iterable[str]) -> set[datetime]:
    timestamps: set[datetime] = set()
    for name in assets:
        match = RADAR_ASSET_RE.match(name)
        if match:
            timestamps.add(parse_observation_timestamp(match.group(1)))
    return timestamps


def gather_release_assets(
    current: ReleaseInfo,
    previous: ReleaseInfo | None,
) -> tuple[dict[str, dict], dict[str, dict]]:
    radar_assets: dict[str, dict] = {}
    phase_assets: dict[str, dict] = {}
    for release in (previous, current):
        if release is None:
            continue
        for name, asset in release.assets.items():
            if RADAR_ASSET_RE.match(name):
                radar_assets[name] = asset
            elif PHASE_ASSET_RE.match(name):
                phase_assets[name] = asset
    return radar_assets, phase_assets


def cleanup_old_releases(store: GitHubReleaseStore, today: datetime) -> None:
    keep_dates = {
        today.date(),
        (today - timedelta(days=1)).date(),
    }
    releases = store.list_history_releases()
    for release in releases:
        tag = str(release.get("tag_name", ""))
        try:
            release_date = datetime.strptime(
                tag.removeprefix("mrms-"), "%Y%m%d"
            ).date()
        except ValueError:
            continue
        if release_date in keep_dates:
            continue
        print(f"  Removing expired history release {tag}")
        release_id = int(release["id"])
        store.delete_release(release_id)
        try:
            store.delete_tag_ref(tag)
        except Exception as exc:
            print(f"    Warning: could not remove tag {tag}: {exc}")


def choose_phase_asset(
    radar_time: datetime,
    phase_assets: dict[str, dict],
) -> dict | None:
    candidates: list[tuple[datetime, dict]] = []
    for name, asset in phase_assets.items():
        match = PHASE_ASSET_RE.match(name)
        if not match:
            continue
        bucket = datetime.strptime(match.group(1), "%Y%m%d-%H%M").replace(
            tzinfo=timezone.utc
        )
        if bucket <= radar_time:
            candidates.append((bucket, asset))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    bucket, asset = candidates[-1]
    return {
        "timestamp_utc": bucket.isoformat(),
        "asset_name": asset["name"],
        "url": asset["browser_download_url"],
        "api_url": asset.get("url"),
    }


def build_manifest(
    observations: list[Observation],
    radar_assets: dict[str, dict],
    phase_assets: dict[str, dict],
    now: datetime,
    bounds: tuple[float, float, float, float],
    releases: list[ReleaseInfo],
) -> dict:
    cutoff = now - timedelta(hours=HISTORY_HOURS)
    usable = [
        obs for obs in observations if cutoff <= obs.valid_time <= now + timedelta(minutes=1)
    ]
    usable.sort(key=lambda item: item.valid_time)

    frames: list[dict] = []
    for obs in usable:
        asset = radar_assets.get(obs.asset_name)
        if asset is None:
            continue
        phase = choose_phase_asset(obs.valid_time, phase_assets)
        frames.append(
            {
                "timestamp_utc": obs.valid_time.isoformat(),
                "radar_url": asset["browser_download_url"],
                "radar_asset": asset["name"],
                "radar_api_url": asset.get("url"),
                "phase_url": phase["url"] if phase else None,
                "phase_asset": phase["asset_name"] if phase else None,
                "phase_api_url": phase.get("api_url") if phase else None,
                "phase_timestamp_utc": phase["timestamp_utc"] if phase else None,
            }
        )

    releases_out = [
        {
            "tag": release.tag,
            "url": release.html_url,
        }
        for release in releases
    ]

    return {
        "version": "1.0-history",
        "generated_at_utc": now.isoformat(),
        "history_hours": HISTORY_HOURS,
        "frame_interval_note": "MRMS ReflectivityAtLowestAltitude observations are timestamped upstream and may be roughly 2 minutes apart.",
        "bounds": [bounds[0], bounds[1], bounds[2], bounds[3]],
        "bounds_format": ["south", "west", "north", "east"],
        "frame_count": len(frames),
        "phase_snapshot_count": len(phase_assets),
        "releases": releases_out,
        "frames": frames,
    }


def run_archive() -> None:
    print("=" * 72)
    print("WINTER RADAR — MRMS 24-HOUR HISTORY ARCHIVE")
    print("=" * 72)

    now = utc_now()
    cutoff = now - timedelta(hours=HISTORY_HOURS)

    token = os.environ.get("GITHUB_TOKEN", "")
    repository = os.environ.get("GITHUB_REPOSITORY", "")
    store = GitHubReleaseStore(token, repository)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    observations = fetch_mrms_directory(session)
    recent_observations = [
        obs for obs in observations
        if cutoff <= obs.valid_time <= now + timedelta(minutes=1)
    ]
    print(
        f"  MRMS observations in previous {HISTORY_HOURS} hours: "
        f"{len(recent_observations)}"
    )

    current_tag = f"mrms-{now:%Y%m%d}"
    previous_day = now - timedelta(days=1)
    previous_tag = f"mrms-{previous_day:%Y%m%d}"

    current_release = store.ensure_release(current_tag)
    previous_release = store.get_release(previous_tag) if previous_tag != current_tag else None

    # Refresh current assets after release creation.
    current_release = store.get_release(current_tag) or current_release

    radar_assets, phase_assets = gather_release_assets(
        current_release,
        previous_release,
    )
    print(f"  Archived radar assets already present: {len(radar_assets)}")
    print(f"  Archived phase snapshots already present: {len(phase_assets)}")

    lats, lons = load_source_coordinates()
    crop = crop_indices(lats, lons, read_history_bounds(OUTPUT_DIR / "mrms_current.json"))
    y0, y1, x0, x1, crop_bounds = crop
    expected_shape = (int(lats.size), int(lons.size))
    row_map_cache: dict[tuple[int, float, float], tuple[int, np.ndarray]] = {}

    print(
        "  History crop: "
        f"rows {y0}:{y1}, cols {x0}:{x1} "
        f"({y1-y0}x{x1-x0})"
    )

    # --------------------------------------------------------------
    # Radar observations
    # --------------------------------------------------------------
    existing_times = existing_radar_timestamps(radar_assets)
    missing = [obs for obs in recent_observations if obs.valid_time not in existing_times]
    missing.sort(key=lambda item: item.valid_time)

    if missing:
        print(
            f"  Missing radar frames to archive: {len(missing)} "
            f"(processing up to {MAX_NEW_RADAR_FRAMES_PER_RUN} this run)"
        )
    else:
        print("  No missing radar frames detected.")

    archived_this_run = 0
    for obs in missing[:MAX_NEW_RADAR_FRAMES_PER_RUN]:
        try:
            data = download_and_render_observation(
                session,
                obs,
                expected_shape,
                (y0, y1, x0, x1),
                crop_bounds,
                row_map_cache,
            )
            upload_release = current_release if obs.valid_time.date() == now.date() else (
                previous_release or current_release
            )
            asset = store.upload_asset(upload_release, obs.asset_name, data)
            target = radar_assets
            target[asset["name"]] = asset
            archived_this_run += 1
            print(
                f"      Uploaded {asset['name']} "
                f"({len(data):,} bytes)"
            )
        except Exception as exc:
            # A single bad historical frame should not suppress the remaining
            # backlog; the next workflow run will retry it. Always print the
            # exception type, and include the full traceback for diagnosis.
            print(
                f"      WARNING: unable to archive {obs.asset_name}: "
                f"{type(exc).__name__}: {exc}"
            )
            if HISTORY_DEBUG:
                import traceback
                traceback.print_exc()

    # --------------------------------------------------------------
    # Phase snapshot: one 10-minute bucket per interval.
    # --------------------------------------------------------------
    phase_bucket = phase_bucket_for_timestamp(now)
    phase_asset_name = f"phase_{phase_bucket:%Y%m%d-%H%M}.webp"
    if phase_asset_name not in phase_assets and (OUTPUT_DIR / "winter_phase_mask.png").exists():
        try:
            data = phase_png_to_webp(
                OUTPUT_DIR / "winter_phase_mask.png",
                (y0, y1, x0, x1),
                crop_bounds,
                row_map_cache,
            )
            asset = store.upload_asset(current_release, phase_asset_name, data)
            phase_assets[asset["name"]] = asset
            print(
                f"  Uploaded phase snapshot {asset['name']} "
                f"({len(data):,} bytes)"
            )
        except Exception as exc:
            print(f"  WARNING: phase snapshot failed: {exc}")
    else:
        print(f"  Phase snapshot already present: {phase_asset_name}")

    # Re-fetch release assets so the manifest always reflects successful uploads
    # even when an upload returned data not represented in the local dictionaries.
    current_release = store.get_release(current_tag) or current_release
    previous_release = (
        store.get_release(previous_tag)
        if previous_tag != current_tag
        else None
    )
    radar_assets, phase_assets = gather_release_assets(current_release, previous_release)

    release_list = [current_release]
    if previous_release is not None:
        release_list.append(previous_release)

    manifest = build_manifest(
        observations=observations,
        radar_assets=radar_assets,
        phase_assets=phase_assets,
        now=now,
        bounds=crop_bounds,
        releases=release_list,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MRMS_HISTORY_FILE.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"  History manifest: {MRMS_HISTORY_FILE}")
    print(f"  History frames available: {manifest['frame_count']}")
    print(f"  Phase snapshots available: {manifest['phase_snapshot_count']}")
    print(f"  New radar frames archived this run: {archived_this_run}")

    cleanup_old_releases(store, now)

    print("MRMS HISTORY ARCHIVE COMPLETE")


def self_test() -> None:
    """Pure-Python smoke test used before handing the files to Actions."""
    html = """
    <a href="MRMS_ReflectivityAtLowestAltitude_00.50_20260920-020241.grib2.gz">a</a>
    <a href="MRMS_ReflectivityAtLowestAltitude_00.50_20260920-020439.grib2.gz">b</a>
    <a href="MRMS_ReflectivityAtLowestAltitude.latest.grib2.gz">latest</a>
    """
    obs = parse_mrms_directory(html)
    assert len(obs) == 2
    assert obs[0].timestamp_key == "20260920-020241"
    assert obs[0].asset_name == "radar_20260920-020241.webp"

    bounds = (40.95, -77.50, 46.15, -68.40)
    lats = np.linspace(54.995, 20.005, 3500)
    lons = np.linspace(-129.995, -60.005, 7000)
    crop = crop_indices(lats, lons, bounds)
    y0, y1, x0, x1, crop_bounds = crop
    assert y1 > y0 and x1 > x0

    sample = np.zeros((y1 - y0, x1 - x0, 4), dtype=np.uint8)
    sample[..., 3] = 255
    row_cache: dict[tuple[int, float, float], tuple[int, np.ndarray]] = {}
    projected = subset_to_webmercator(sample, crop_bounds, row_cache)
    assert projected.shape[1] == sample.shape[1]
    assert projected.shape[2] == 4

    payload = rgba_to_webp_bytes(projected, quality=82)
    assert payload[:4] == b"RIFF"

    stamp = floor_time(datetime(2026, 9, 20, 7, 37, tzinfo=timezone.utc), 10)
    assert stamp.minute == 30

    print("MRMS HISTORY SELF-TEST PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        run_archive()
