from __future__ import annotations

"""Archive timestamped MRMS observations and phase masks for WinterRadar.

The live WinterRadar pipeline intentionally continues to use the `latest`
MRMS product. This module is a secondary history collector. Each time the
GitHub Actions workflow runs it:

1. Reads NOAA's timestamped MergedReflectivityQCComposite directory.
2. Finds all observations from the previous 24 hours.
3. Looks at the two daily GitHub Releases used by WinterRadar as persistent
   storage and determines which observations are missing.
4. Downloads and renders only those missing observations.
5. Stores native-resolution WebP frames in the appropriate daily release.
6. Archives one full-CONUS winter-phase mask and one full-CONUS precipitation-
   type composite per 10-minute bucket, keeping the history comfortably below
   GitHub's 1,000-asset-per-release limit.
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
    "https://mrms.ncep.noaa.gov/2D/MergedReflectivityQCComposite/"
)

MRMS_FILENAME_RE = re.compile(
    r"MRMS_MergedReflectivityQCComposite_00\.50_(\d{8}-\d{6})\.grib2\.gz"
)

# New namespace prevents the viewer from ever mixing old
# older radar frames with the current QC-composite frames.
RADAR_ASSET_RE = re.compile(r"^radar_qc_conus_(\d{8}-\d{6})\.webp$")
PHASE_ASSET_RE = re.compile(r"^phase_conus_(\d{8}-\d{6}|\d{8}-\d{4})\.webp$")
PRECIP_TYPE_ASSET_RE = re.compile(r"^preciptype_conus_(\d{8}-\d{6}|\d{8}-\d{4})\.webp$")
HISTORY_RELEASE_RE = re.compile(r"^mrms-(\d{8})(?:-(\d+))?$")

# GitHub enforces a hard 1,000-asset maximum per release. Keep the active
# partition below that ceiling so a phase snapshot can never consume the
# final slot and block radar uploads. The normal 8-hour CONUS archive fits comfortably within the release limit,
# but partitioning also recovers cleanly from older releases that already
# reached the limit.
MAX_ASSETS_PER_RELEASE = 950

# Full-CONUS history extent. Recent radar observations are retained for
# 8 hours so the viewer has broad national context without the storage
# cost of a 24-hour full-CONUS archive.
HISTORY_FALLBACK_BOUNDS = [
    [20.005001, -129.995],
    [54.995, -60.005002],
]

# Archive all observed ~2-minute MRMS frames for the most recent 8 hours.
# Every archived radar observation gets its own precipitation-type composite.
# RAP environmental profiles are reused by valid hour; they do not force radar
# observations to wait for a new model cycle.
HISTORY_HOURS = 8
PHASE_BUCKET_MINUTES = 5  # retained only for backwards-compatible old assets
# Keep each archive invocation short enough for the 5-minute workflow.
# Newest observations are always included; only a small number of older
# observations are processed as catch-up work.
MAX_NEW_RADAR_FRAMES_PER_RUN = int(
    os.environ.get("MRMS_HISTORY_MAX_FRAMES_PER_RUN", "6")
)
MAX_NEW_PHASE_FRAMES_PER_RUN = int(
    os.environ.get("MRMS_HISTORY_MAX_PHASE_FRAMES_PER_RUN", "6")
)

REQUEST_TIMEOUT = (20, 120)
UPLOAD_TIMEOUT = (20, 180)
USER_AGENT = "WinterRadar/1.2 (MRMS 8-hour full-CONUS history collector)"
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
        return f"radar_qc_conus_{self.timestamp_key}.webp"


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
            "name": f"WinterRadar MRMS 8-Hour CONUS History — {tag.removeprefix('mrms-')}",
            "body": (
                "Automated 8-hour WinterRadar MRMS CONUS observation archive. "
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
        releases: list[dict] = []
        page = 1
        while True:
            response = self.session.get(
                self._url("/releases"),
                params={"per_page": 100, "page": page},
                timeout=REQUEST_TIMEOUT,
            )
            if not response.ok:
                detail = response.text[:500].replace("\n", " ")
                raise RuntimeError(
                    f"GitHub release listing failed: HTTP {response.status_code}: {detail}"
                )
            batch = response.json()
            releases.extend(
                release
                for release in batch
                if HISTORY_RELEASE_RE.match(str(release.get("tag_name", "")))
            )
            if len(batch) < 100:
                break
            page += 1
        return releases


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
            "No timestamped MergedReflectivityQCComposite observations were found "
            "in the MRMS directory."
        )
    return observations


def read_history_bounds(metadata_path: Path) -> tuple[float, float, float, float]:
    """Return (south, west, north, east) for the full-CONUS history."""
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            live_bounds = metadata.get("bounds")
            if isinstance(live_bounds, list) and len(live_bounds) == 4:
                south, west, north, east = (float(value) for value in live_bounds)
                if south < north and west < east:
                    return south, west, north, east
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


def rgba_to_webp_bytes(
    rgba: np.ndarray,
    quality: int = 90,
    max_width: int | None = None,
) -> bytes:
    """Encode a history frame without spatial downsampling.

    The MRMS source grid is already the native ~1-km product. The previous
    3500-pixel cap used LANCZOS resampling and made archived radar look soft
    when enlarged in the browser. History now preserves the native pixel
    grid; browser CSS handles pixel-preserving enlargement.
    """
    image = Image.fromarray(np.asarray(rgba, dtype=np.uint8), mode="RGBA")
    if max_width and image.width > max_width:
        new_height = max(1, round(image.height * max_width / image.width))
        image = image.resize((max_width, new_height), Image.Resampling.NEAREST)

    buffer = io.BytesIO()
    image.save(
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
    keep_grib: bool = False,
) -> tuple[bytes, Path | None]:
    """Download one MRMS frame, render WebP, and optionally preserve the GRIB for per-scan phase processing."""
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
        if keep_grib:
            # Give the phase helper a deterministic timestamped filename.
            kept = Path(tempfile.gettempdir()) / f"MRMS_{observation.timestamp_key}.grib2"
            kept.unlink(missing_ok=True)
            grib_path.replace(kept)
            return data, kept
        return data, None

    finally:
        gz_path.unlink(missing_ok=True)
        if not keep_grib:
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


def history_release_date(tag: str) -> object | None:
    match = HISTORY_RELEASE_RE.match(str(tag))
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y%m%d").date()


def history_release_sort_key(tag: str) -> tuple[int, int]:
    match = HISTORY_RELEASE_RE.match(str(tag))
    if not match:
        return (0, 0)
    return (int(match.group(1)), int(match.group(2) or 0))


def gather_release_assets_many(
    releases: Iterable[ReleaseInfo],
) -> tuple[dict[str, dict], dict[str, dict], dict[str, dict]]:
    radar_assets: dict[str, dict] = {}
    phase_assets: dict[str, dict] = {}
    precip_type_assets: dict[str, dict] = {}
    for release in releases:
        for name, asset in release.assets.items():
            if RADAR_ASSET_RE.match(name):
                radar_assets[name] = asset
            elif PHASE_ASSET_RE.match(name):
                phase_assets[name] = asset
            elif PRECIP_TYPE_ASSET_RE.match(name):
                precip_type_assets[name] = asset
    return radar_assets, phase_assets, precip_type_assets


def release_tag_for_date(date_value, part: int) -> str:
    base = f"mrms-{date_value:%Y%m%d}"
    return base if part == 0 else f"{base}-{part:02d}"


def ensure_upload_release(
    store: GitHubReleaseStore,
    date_value,
    release_index: dict[object, list[ReleaseInfo]],
) -> ReleaseInfo:
    candidates = release_index.setdefault(date_value, [])
    candidates.sort(key=lambda rel: history_release_sort_key(rel.tag), reverse=True)

    for release in candidates:
        if len(release.assets) < MAX_ASSETS_PER_RELEASE:
            return release

    used_parts = []
    for release in candidates:
        match = HISTORY_RELEASE_RE.match(release.tag)
        if match:
            used_parts.append(int(match.group(2) or 0))
    next_part = max(used_parts, default=-1) + 1

    tag = release_tag_for_date(date_value, next_part)
    release = store.get_release(tag)
    if release is None:
        release = store.create_release(tag)
    candidates.append(release)
    candidates.sort(key=lambda rel: history_release_sort_key(rel.tag), reverse=True)

    print(
        f"  Using history release partition {release.tag} "
        f"({len(release.assets)} assets before upload)"
    )
    return release


def cleanup_old_releases(store: GitHubReleaseStore, today: datetime) -> None:
    keep_dates = {
        today.date(),
        (today - timedelta(days=1)).date(),
    }
    releases = store.list_history_releases()
    for release in releases:
        tag = str(release.get("tag_name", ""))
        release_date = history_release_date(tag)
        if release_date is None:
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
    exact_name = f"phase_conus_{radar_time:%Y%m%d-%H%M%S}.webp"
    exact = phase_assets.get(exact_name)
    if exact is not None:
        return {
            "timestamp_utc": radar_time.isoformat(),
            "asset_name": exact["name"],
            "url": exact["browser_download_url"],
            "api_url": exact.get("url"),
        }

    candidates: list[tuple[datetime, dict]] = []
    for name, asset in phase_assets.items():
        match = PHASE_ASSET_RE.match(name)
        if not match:
            continue
        stamp = match.group(1)
        fmt = "%Y%m%d-%H%M%S" if len(stamp) == 15 else "%Y%m%d-%H%M"
        bucket = datetime.strptime(stamp, fmt).replace(tzinfo=timezone.utc)
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


def choose_precip_type_asset(
    radar_time: datetime,
    precip_type_assets: dict[str, dict],
) -> dict | None:
    # New per-scan assets use the exact radar timestamp.  This is the primary
    # path and prevents the history slider from pairing a radar frame with an
    # older 5-minute composite.
    exact_name = f"preciptype_conus_{radar_time:%Y%m%d-%H%M%S}.webp"
    exact = precip_type_assets.get(exact_name)
    if exact is not None:
        return {
            "timestamp_utc": radar_time.isoformat(),
            "asset_name": exact["name"],
            "url": exact["browser_download_url"],
            "api_url": exact.get("url"),
        }

    # Backward compatibility for history generated before the per-scan change.
    candidates: list[tuple[datetime, dict]] = []
    for name, asset in precip_type_assets.items():
        match = PRECIP_TYPE_ASSET_RE.match(name)
        if not match:
            continue
        stamp = match.group(1)
        fmt = "%Y%m%d-%H%M%S" if len(stamp) == 15 else "%Y%m%d-%H%M"
        bucket = datetime.strptime(stamp, fmt).replace(tzinfo=timezone.utc)
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
    precip_type_assets: dict[str, dict],
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
        precip_type = choose_precip_type_asset(obs.valid_time, precip_type_assets)
        # Backward-compatible fallback: older history releases used the phase
        # asset slot for the composite. New snapshots are stored separately.
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
                "precip_type_url": precip_type["url"] if precip_type else None,
                "precip_type_asset": precip_type["asset_name"] if precip_type else None,
                "precip_type_api_url": precip_type.get("api_url") if precip_type else None,
                "precip_type_timestamp_utc": precip_type["timestamp_utc"] if precip_type else None,
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
        "frame_interval_note": "MRMS MergedReflectivityQCComposite observations are timestamped upstream and may be roughly 2 minutes apart. The archive retains the most recent 8 hours at full-CONUS coverage.",
        "bounds": [bounds[0], bounds[1], bounds[2], bounds[3]],
        "bounds_format": ["south", "west", "north", "east"],
        "frame_count": len(frames),
        "phase_snapshot_count": len(phase_assets),
        "precip_type_snapshot_count": len(precip_type_assets),
        "releases": releases_out,
        "frames": frames,
    }


def run_archive() -> None:
    print("=" * 72)
    print("WINTER RADAR — MRMS 8-HOUR MERGED COMPOSITE HISTORY ARCHIVE")
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

    previous_day = now - timedelta(days=1)
    history_dates = {now.date(), previous_day.date()}

    # Load every MRMS history release partition covering the two dates in the
    # rolling 8-hour window. A date may have a base release plus one or more
    # overflow partitions once the 1,000-asset GitHub limit is reached.
    release_index: dict[object, list[ReleaseInfo]] = {date_value: [] for date_value in history_dates}
    for meta in store.list_history_releases():
        tag = str(meta.get("tag_name", ""))
        release_date = history_release_date(tag)
        if release_date not in history_dates:
            continue
        release = store.get_release(tag)
        if release is not None:
            release_index.setdefault(release_date, []).append(release)

    # Establish the standard daily release when it does not exist. If it is
    # already full, ensure_upload_release will automatically create an overflow
    # partition instead of attempting another upload into a 1,000-asset release.
    if not release_index[now.date()]:
        base = store.ensure_release(f"mrms-{now:%Y%m%d}")
        release_index[now.date()].append(base)
    if not release_index[previous_day.date()]:
        base = store.get_release(f"mrms-{previous_day:%Y%m%d}")
        if base is not None:
            release_index[previous_day.date()].append(base)

    all_history_releases: list[ReleaseInfo] = []
    for releases_for_date in release_index.values():
        all_history_releases.extend(releases_for_date)
    all_history_releases.sort(key=lambda rel: history_release_sort_key(rel.tag))

    radar_assets, phase_assets, precip_type_assets = gather_release_assets_many(all_history_releases)
    print(f"  History release partitions available: {len(all_history_releases)}")
    for date_value in sorted(release_index):
        labels = [f"{rel.tag} ({len(rel.assets)})" for rel in sorted(release_index[date_value], key=lambda rel: history_release_sort_key(rel.tag))]
        if labels:
            print("  " + ", ".join(labels))
    print(f"  Archived radar assets already present: {len(radar_assets)}")
    print(f"  Archived phase snapshots already present: {len(phase_assets)}")
    print(f"  Archived precipitation-type snapshots already present: {len(precip_type_assets)}")

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
        # Keep the history viewer current while also backfilling the 8-hour
        # archive. The per-run cap is intentionally small: a full-CONUS
        # 3500x7000 decode is expensive, and the workflow runs every 5 min. A pure oldest-first queue can leave the viewer many hours
        # behind while the initial backlog is being filled. Split each run
        # between the oldest and newest missing observations so the right edge
        # of the slider stays close to the current MRMS scan.
        limit = max(1, MAX_NEW_RADAR_FRAMES_PER_RUN)
        if len(missing) > limit:
            oldest_count = limit // 2
            newest_count = limit - oldest_count
            selected = missing[:oldest_count] + missing[-newest_count:]
            selected.sort(key=lambda item: item.valid_time)
        else:
            selected = missing

        print(
            f"  Missing radar frames to archive: {len(missing)} "
            f"(processing {len(selected)} this run; newest frames prioritized)"
        )
    else:
        selected = []
        print("  No missing radar frames detected.")

    archived_this_run = 0
    phase_grib_paths: list[Path] = []
    phase_observations: list[Observation] = []

    # Phase/composite backfill is independent of radar backfill. This matters
    # after deploying the per-scan change: radar frames may already exist in
    # GitHub Releases but only have the old 5-minute phase association.
    exact_phase_missing = []
    for obs in recent_observations:
        exact_phase_name = f"phase_conus_{obs.timestamp_key}.webp"
        exact_precip_name = f"preciptype_conus_{obs.timestamp_key}.webp"
        if exact_phase_name not in phase_assets or exact_precip_name not in precip_type_assets:
            exact_phase_missing.append(obs)
    exact_phase_missing.sort(key=lambda item: item.valid_time)
    selected_phase = {obs.valid_time for obs in selected}
    extra_phase = [obs for obs in exact_phase_missing if obs.valid_time not in selected_phase]
    if len(extra_phase) > MAX_NEW_PHASE_FRAMES_PER_RUN:
        # Backfill only a small tail each run. Selected radar observations
        # are always processed for exact per-scan phase products.
        extra_phase = extra_phase[-MAX_NEW_PHASE_FRAMES_PER_RUN:]
    phase_work_observations = list(selected) + extra_phase
    phase_work_observations.sort(key=lambda item: item.valid_time)
    print(
        f"  Per-scan phase/composite work items: {len(phase_work_observations)} "
        f"({len(extra_phase)} backfill-only; max {MAX_NEW_PHASE_FRAMES_PER_RUN} backfill/run)"
    )

    for obs in phase_work_observations:
        try:
            data, grib_path = download_and_render_observation(
                session,
                obs,
                expected_shape,
                (y0, y1, x0, x1),
                crop_bounds,
                row_map_cache,
                keep_grib=True,
            )
            if grib_path is not None:
                phase_grib_paths.append(grib_path)
                phase_observations.append(obs)
            if obs.valid_time in {item.valid_time for item in selected}:
                upload_release = ensure_upload_release(
                    store,
                    obs.valid_time.date(),
                    release_index,
                )
                asset = store.upload_asset(upload_release, obs.asset_name, data)
                upload_release.assets[asset["name"]] = asset
                radar_assets[asset["name"]] = asset
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
    # Per-scan phase/composite generation.
    # Each radar observation gets a matching phase and precipitation-type
    # asset. RAP environmental profiles are cached by valid hour inside the
    # isolated helper process, so hourly model timing never skips a radar scan.
    # --------------------------------------------------------------
    if phase_grib_paths:
        helper_output = Path(tempfile.mkdtemp(prefix="winterradar_phase_"))
        try:
            import subprocess
            helper = ROOT / "src" / "build_history_precip_type.py"
            if not helper.exists():
                helper = ROOT / "build_history_precip_type.py"
            cmd = ["python", str(helper), "--output-dir", str(helper_output)] + [str(path) for path in phase_grib_paths]
            print(f"  Building per-scan phase composites for {len(phase_grib_paths)} radar scans...")
            subprocess.run(cmd, check=True)

            for obs in phase_observations:
                names = [
                    f"phase_conus_{obs.timestamp_key}.webp",
                    f"preciptype_conus_{obs.timestamp_key}.webp",
                ]
                for name in names:
                    source = helper_output / name
                    if not source.exists():
                        print(f"  WARNING: per-scan phase asset missing: {name}")
                        continue
                    if name.startswith("phase_conus_"):
                        target_assets = phase_assets
                    else:
                        target_assets = precip_type_assets
                    if name in target_assets:
                        continue
                    data = source.read_bytes()
                    release = ensure_upload_release(store, obs.valid_time.date(), release_index)
                    asset = store.upload_asset(release, name, data)
                    release.assets[asset["name"]] = asset
                    target_assets[asset["name"]] = asset
                    print(f"  Uploaded per-scan phase asset {name} ({len(data):,} bytes)")
        except Exception as exc:
            print(f"  WARNING: per-scan phase generation failed: {type(exc).__name__}: {exc}")
        finally:
            for path in phase_grib_paths:
                path.unlink(missing_ok=True)
            shutil.rmtree(helper_output, ignore_errors=True)

    # --------------------------------------------------------------
    # Phase + precipitation-type snapshots: one 10-minute bucket per interval.
    # --------------------------------------------------------------
    snapshot_bucket = phase_bucket_for_timestamp(now)
    phase_asset_name = f"phase_conus_{snapshot_bucket:%Y%m%d-%H%M}.webp"
    precip_type_asset_name = f"preciptype_conus_{snapshot_bucket:%Y%m%d-%H%M}.webp"

    phase_source = OUTPUT_DIR / "winter_phase_mask.png"
    precip_type_source = OUTPUT_DIR / "winter_precip_type.png"

    if phase_asset_name not in phase_assets and phase_source.exists():
        try:
            data = phase_png_to_webp(
                phase_source,
                (y0, y1, x0, x1),
                crop_bounds,
                row_map_cache,
            )
            phase_release = ensure_upload_release(
                store,
                now.date(),
                release_index,
            )
            asset = store.upload_asset(phase_release, phase_asset_name, data)
            phase_release.assets[asset["name"]] = asset
            phase_assets[asset["name"]] = asset
            print(
                f"  Uploaded phase snapshot {asset['name']} "
                f"({len(data):,} bytes)"
            )
        except Exception as exc:
            print(f"  WARNING: phase snapshot failed: {exc}")
    else:
        print(f"  Phase snapshot already present: {phase_asset_name}")

    if precip_type_asset_name not in precip_type_assets and precip_type_source.exists():
        try:
            data = phase_png_to_webp(
                precip_type_source,
                (y0, y1, x0, x1),
                crop_bounds,
                row_map_cache,
            )
            precip_release = ensure_upload_release(
                store,
                now.date(),
                release_index,
            )
            asset = store.upload_asset(precip_release, precip_type_asset_name, data)
            precip_release.assets[asset["name"]] = asset
            precip_type_assets[asset["name"]] = asset
            print(
                f"  Uploaded precipitation-type snapshot {asset['name']} "
                f"({len(data):,} bytes)"
            )
        except Exception as exc:
            print(f"  WARNING: precipitation-type snapshot failed: {exc}")
    else:
        print(f"  Precipitation-type snapshot already present: {precip_type_asset_name}")

    # Re-fetch all active release partitions so the manifest reflects every
    # successful upload, including any overflow partition created above.
    refreshed_releases: list[ReleaseInfo] = []
    for date_value in history_dates:
        refreshed_for_date: list[ReleaseInfo] = []
        for release in release_index.get(date_value, []):
            refreshed = store.get_release(release.tag)
            if refreshed is not None:
                refreshed_for_date.append(refreshed)
        release_index[date_value] = refreshed_for_date
        refreshed_releases.extend(refreshed_for_date)

    refreshed_releases.sort(key=lambda rel: history_release_sort_key(rel.tag))
    radar_assets, phase_assets, precip_type_assets = gather_release_assets_many(refreshed_releases)
    release_list = refreshed_releases

    manifest = build_manifest(
        observations=observations,
        radar_assets=radar_assets,
        phase_assets=phase_assets,
        precip_type_assets=precip_type_assets,
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
    print(f"  Precipitation-type snapshots available: {manifest['precip_type_snapshot_count']}")
    print(f"  New radar frames archived this run: {archived_this_run}")

    cleanup_old_releases(store, now)

    print("MRMS HISTORY ARCHIVE COMPLETE")


def self_test() -> None:
    """Pure-Python smoke test used before handing the files to Actions."""
    html = """
    <a href="MRMS_MergedReflectivityQCComposite_00.50_20260920-020241.grib2.gz">a</a>
    <a href="MRMS_MergedReflectivityQCComposite_00.50_20260920-020439.grib2.gz">b</a>
    <a href="MRMS_MergedReflectivityQCComposite.latest.grib2.gz">latest</a>
    """
    obs = parse_mrms_directory(html)
    assert len(obs) == 2
    assert obs[0].timestamp_key == "20260920-020241"
    assert obs[0].asset_name == "radar_qc_conus_20260920-020241.webp"
    assert PRECIP_TYPE_ASSET_RE.match("preciptype_conus_20260920-0230.webp")

    bounds = (20.005001, -129.995, 54.995, -60.005002)
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

    stamp = floor_time(datetime(2026, 9, 20, 7, 37, tzinfo=timezone.utc), 5)
    assert stamp.minute == 35

    print("MRMS HISTORY SELF-TEST PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        run_archive()
