#!/usr/bin/env python3
"""Stage persistent MRMS Release history into the GitHub Pages tree.

GitHub Actions runners are ephemeral, so historical MRMS files live in GitHub
Release assets.  This script reads the history manifest from the Release
partitions, downloads the referenced WebP assets, and places them in
site_history_tmp for the Pages build.
"""

from __future__ import annotations

import concurrent.futures
import io
import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote
import re

from PIL import Image
import requests

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"
HISTORY_DIR = ROOT / "site_history_tmp"
TOKEN = os.environ.get("GITHUB_TOKEN", "")
REPO = os.environ.get("GITHUB_REPOSITORY", "")
API = "https://api.github.com"
API_VERSION = "2026-03-10"
UA = "WinterRadar/1.5 (Pages history staging)"

# History WebPs in the persistent Release store use the full-CONUS Web Mercator
# geometry. Pages only needs the Northeast browser domain, so every staged
# history frame is cropped to the same regional bounds used by the live viewer.
PAGES_HISTORY_BOUNDS = (37.0, -84.5, 48.5, -66.0)
DEFAULT_FULL_BOUNDS = (20.005001, -129.995, 54.995, -60.00500199999999)
WEBP_QUALITY = 90

def mercator_y(lat_deg: float) -> float:
    lat = max(-85.0511287798, min(85.0511287798, float(lat_deg)))
    return math.log(math.tan(math.pi / 4.0 + math.radians(lat) / 2.0))


def crop_history_asset_to_pages(data: bytes, full_bounds: tuple[float, float, float, float]) -> bytes:
    """Crop a full-CONUS Web Mercator history raster to the Pages NEUS domain."""
    south, west, north, east = full_bounds
    tsouth, twest, tnorth, teast = PAGES_HISTORY_BOUNDS

    if not (south < tsouth < tnorth < north and west < twest < teast < east):
        full_bounds = DEFAULT_FULL_BOUNDS
        south, west, north, east = full_bounds

    with Image.open(io.BytesIO(data)) as im:
        image = im.convert("RGBA")
        width, height = image.size

        full_y_s = mercator_y(south)
        full_y_n = mercator_y(north)
        full_y_span = full_y_n - full_y_s

        x0 = int(math.floor((twest - west) / (east - west) * width))
        x1 = int(math.ceil((teast - west) / (east - west) * width))
        y0 = int(math.floor((full_y_n - mercator_y(tnorth)) / full_y_span * height))
        y1 = int(math.ceil((full_y_n - mercator_y(tsouth)) / full_y_span * height))

        x0 = max(0, min(width - 1, x0))
        x1 = max(x0 + 1, min(width, x1))
        y0 = max(0, min(height - 1, y0))
        y1 = max(y0 + 1, min(height, y1))

        cropped = image.crop((x0, y0, x1, y1))

    buffer = io.BytesIO()
    cropped.save(
        buffer,
        format="WEBP",
        quality=WEBP_QUALITY,
        method=6,
    )
    return buffer.getvalue()


def api_url(path: str) -> str:
    return f"{API}/repos/{REPO}{path}"


def headers(accept: str = "application/vnd.github+json") -> dict[str, str]:
    return {
        "Accept": accept,
        "Authorization": f"Bearer {TOKEN}",
        "X-GitHub-Api-Version": API_VERSION,
        "User-Agent": UA,
    }


def list_releases(session: requests.Session) -> list[dict]:
    """Return recent MRMS releases, including all partitioned releases."""
    releases: list[dict] = []
    page = 1
    while True:
        r = session.get(
            api_url("/releases"),
            headers=headers(),
            params={"per_page": 100, "page": page},
            timeout=(20, 60),
        )
        r.raise_for_status()
        batch = r.json()
        releases.extend(batch)
        if len(batch) < 100:
            return releases
        page += 1


def list_assets(session: requests.Session, release_id: int) -> dict[str, dict]:
    assets: dict[str, dict] = {}
    page = 1
    while True:
        r = session.get(
            api_url(f"/releases/{release_id}/assets"),
            headers=headers(),
            params={"per_page": 100, "page": page},
            timeout=(20, 60),
        )
        r.raise_for_status()
        batch = r.json()
        for item in batch:
            assets[str(item["name"])] = item
        if len(batch) < 100:
            return assets
        page += 1


def download_asset(session: requests.Session, asset: dict) -> bytes:
    r = session.get(
        asset["url"],
        headers=headers("application/octet-stream"),
        timeout=(20, 180),
    )
    r.raise_for_status()
    return r.content


RADAR_ASSET_RE = re.compile(r"^radar_qc_conus_(\d{8}-\d{6})\.webp$")


def parse_radar_asset_timestamp(name: str) -> datetime | None:
    match = RADAR_ASSET_RE.match(name)
    if not match:
        return None
    try:
        return datetime.strptime(
            match.group(1), "%Y%m%d-%H%M%S"
        ).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def empty_manifest() -> dict:
    return {
        "version": "1.0-history",
        "history_hours": 3,
        "frame_count": 0,
        "phase_snapshot_count": 0,
        "precip_type_snapshot_count": 0,
        "frames": [],
    }


def release_sort_key(release: dict) -> tuple[str, int]:
    tag = str(release.get("tag_name", ""))
    # Base release sorts before -01/-02. We primarily use updated_at as a
    # tiebreaker so the most recently updated partition is considered first.
    suffix = 0
    if "-" in tag:
        try:
            suffix = int(tag.rsplit("-", 1)[1])
        except ValueError:
            suffix = 0
    return (str(release.get("updated_at", "")), suffix)


def main() -> None:
    if not TOKEN or "/" not in REPO:
        raise SystemExit("GITHUB_TOKEN/GITHUB_REPOSITORY are required")

    session = requests.Session()

    now = datetime.now(timezone.utc)
    dates = [now.date(), (now - timedelta(days=1)).date()]
    wanted_prefixes = {f"mrms-{d:%Y%m%d}" for d in dates}

    print("=" * 68)
    print("WINTER RADAR — PAGES HISTORY STAGING")
    print("=" * 68)
    print("  Reading MRMS history from persistent GitHub Release partitions")

    all_releases = list_releases(session)
    candidates = [
        r for r in all_releases
        if any(
            str(r.get("tag_name", "")) == prefix
            or str(r.get("tag_name", "")).startswith(prefix + "-")
            for prefix in wanted_prefixes
        )
    ]
    candidates.sort(key=release_sort_key, reverse=True)

    if not candidates:
        print("  No MRMS history release partitions exist yet; Pages will publish without history.")
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUT / "mrms_history.json").write_text(
            json.dumps(empty_manifest(), indent=2), encoding="utf-8"
        )
        return

    # Build one combined asset index across every partition. This matters once
    # a day's release exceeds GitHub's practical asset-count target and is
    # split into mrms-YYYYMMDD, mrms-YYYYMMDD-01, etc.
    release_assets: list[tuple[dict, dict[str, dict]]] = []
    combined_assets: dict[str, dict] = {}
    manifest_candidates: list[tuple[dict, dict, dict]] = []

    for release in candidates:
        tag = release.get("tag_name", "")
        assets = list_assets(session, int(release["id"]))
        release_assets.append((release, assets))
        combined_assets.update(assets)

        manifest_asset = assets.get("mrms_history.json")
        if manifest_asset:
            try:
                manifest = json.loads(download_asset(session, manifest_asset).decode("utf-8"))
                manifest_candidates.append((release, assets, manifest))
                print(
                    f"  Found manifest in {tag}: "
                    f"{manifest.get('frame_count', len(manifest.get('frames', [])))} frames"
                )
            except Exception as exc:
                print(f"  WARNING: could not read manifest from {tag}: {exc}")

    if not manifest_candidates:
        print("  No mrms_history.json found in any current Release partition; Pages will publish without history.")
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUT / "mrms_history.json").write_text(
            json.dumps(empty_manifest(), indent=2), encoding="utf-8"
        )
        return

    # Do NOT select a single partition manifest. Each GitHub Release partition
    # can have its own partial rolling manifest, so choosing the manifest with
    # the largest frame count can make Pages jump backward to an older chunk
    # when the newest daily release contains fewer frames.
    #
    # Merge every manifest, de-duplicate by timestamp, then keep the newest
    # rolling window. The asset index above spans all partitions, so every
    # merged frame can still be downloaded regardless of which release holds it.
    merged_by_timestamp: dict[str, dict] = {}
    bounds_candidates: list[tuple[str, list[float]]] = []

    for release, _, manifest_part in manifest_candidates:
        tag = str(release.get("tag_name", ""))
        bounds = manifest_part.get("bounds")
        if isinstance(bounds, list) and len(bounds) == 4:
            try:
                parsed = [float(v) for v in bounds]
                if all(map(lambda v: v == v, parsed)):
                    bounds_candidates.append((
                        str(release.get("updated_at", "")),
                        parsed,
                    ))
            except (TypeError, ValueError):
                pass

        for frame in manifest_part.get("frames", []):
            if not isinstance(frame, dict):
                continue
            ts = str(frame.get("timestamp_utc", "")).strip()
            if not ts:
                continue

            # Prefer the newer copy when the same observation exists in more
            # than one partition, but NEVER let a newer partial/live manifest
            # erase a composite that an older archive manifest already has.
            # Live buffering intentionally publishes radar-only frames while
            # composites are still being backfilled.
            existing = merged_by_timestamp.get(ts)
            if existing is None:
                merged_by_timestamp[ts] = dict(frame)
            else:
                merged = dict(existing)
                for key, value in frame.items():
                    if value is None or value == "":
                        # Preserve a populated value from another manifest.
                        if merged.get(key) not in (None, ""):
                            continue
                    merged[key] = value
                merged_by_timestamp[ts] = merged

    merged_frames = list(merged_by_timestamp.values())

    def frame_time(frame: dict) -> datetime | None:
        try:
            return datetime.fromisoformat(str(frame.get("timestamp_utc", "")).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None

    merged_frames = [
        frame for frame in merged_frames
        if frame_time(frame) is not None
    ]
    merged_frames.sort(key=lambda frame: frame_time(frame) or datetime.min.replace(tzinfo=timezone.utc))

    # Only add a new radar timestamp directly when the complete matching
    # phase and precipitation-type assets are already present in the Release
    # asset index. Never add radar-only frames to the Pages timeline: those
    # create an apparent "blank" timestep in the default composite view.
    manifest_timestamps = {
        frame_time(frame)
        for frame in merged_frames
        if frame_time(frame) is not None
    }

    now = datetime.now(timezone.utc)
    live_cutoff = now - timedelta(hours=3)

    for asset_name, asset in combined_assets.items():
        radar_time = parse_radar_asset_timestamp(asset_name)
        if radar_time is None:
            continue
        if radar_time < live_cutoff or radar_time > now + timedelta(minutes=1):
            continue
        if radar_time in manifest_timestamps:
            continue

        stamp = radar_time.strftime("%Y%m%d-%H%M%S")
        phase_name = f"phase_conus_{stamp}.webp"
        precip_name = f"preciptype_conus_{stamp}.webp"
        phase_asset = combined_assets.get(phase_name)
        precip_asset = combined_assets.get(precip_name)
        if not phase_asset or not precip_asset:
            continue

        merged_frames.append(
            {
                "timestamp_utc": radar_time.isoformat(),
                "radar_url": asset.get("browser_download_url"),
                "radar_asset": asset_name,
                "radar_api_url": asset.get("url"),
                "phase_url": phase_asset.get("browser_download_url"),
                "phase_asset": phase_name,
                "phase_api_url": phase_asset.get("url"),
                "phase_timestamp_utc": radar_time.isoformat(),
                "precip_type_url": precip_asset.get("browser_download_url"),
                "precip_type_asset": precip_name,
                "precip_type_api_url": precip_asset.get("url"),
                "precip_type_timestamp_utc": radar_time.isoformat(),
            }
        )
        manifest_timestamps.add(radar_time)

    # The operational history timeline is a synchronized product timeline.
    # Keep only frames that have all three assets required to render the
    # default composite without a blank/missing-data slot.
    merged_frames = [
        frame
        for frame in merged_frames
        if frame.get("radar_asset")
        and frame.get("phase_asset")
        and frame.get("precip_type_asset")
    ]

    merged_frames.sort(key=lambda frame: frame_time(frame) or datetime.min.replace(tzinfo=timezone.utc))

    history_hours = 3
    merged_frames = [
        frame
        for frame in merged_frames
        if live_cutoff <= (frame_time(frame) or datetime.min.replace(tzinfo=timezone.utc)) <= now + timedelta(minutes=1)
    ]

    # Build the final manifest from the merged frame set. This guarantees that
    # the newest live frames and older archive frames coexist in one Pages
    # history index, rather than one partition replacing the other.
    radar_names = {
        str(frame.get("radar_asset"))
        for frame in merged_frames
        if frame.get("radar_asset")
    }
    phase_names = {
        str(frame.get("phase_asset"))
        for frame in merged_frames
        if frame.get("phase_asset")
    }
    precip_names = {
        str(frame.get("precip_type_asset"))
        for frame in merged_frames
        if frame.get("precip_type_asset")
    }

    latest_bounds = None
    if bounds_candidates:
        bounds_candidates.sort(key=lambda item: item[0], reverse=True)
        latest_bounds = bounds_candidates[0][1]

    manifest = {
        "version": "1.0-history",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "history_hours": history_hours,
        "frame_interval_note": "MRMS MergedReflectivityQCComposite observations are timestamped upstream and may be roughly 2 minutes apart. Pages combines current Release partitions, retains the synchronized three-hour window, stages compact Northeast browser rasters same-origin, and publishes only frames containing radar + phase + precipitation-type products.",
        "bounds": list(PAGES_HISTORY_BOUNDS),
        "bounds_format": ["south", "west", "north", "east"],
        "source_bounds": latest_bounds or list(DEFAULT_FULL_BOUNDS),
        "frame_count": len(merged_frames),
        "phase_snapshot_count": len(phase_names),
        "precip_type_snapshot_count": len(precip_names),
        "releases": [
            {"tag": r.get("tag_name", ""), "url": r.get("html_url", "")}
            for r, _, _ in manifest_candidates
        ],
        "frames": merged_frames,
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "mrms_history.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    # Use the latest archive-status asset we can find as well.
    status_assets = [
        (r, assets["mrms_archive_status.json"])
        for r, assets in release_assets
        if "mrms_archive_status.json" in assets
    ]
    if status_assets:
        status_assets.sort(key=lambda item: str(item[0].get("updated_at", "")), reverse=True)
        try:
            (OUTPUT / "mrms_archive_status.json").write_bytes(
                download_asset(session, status_assets[0][1])
            )
        except Exception as exc:
            print(f"  WARNING: could not stage archive status: {exc}")

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    for old in HISTORY_DIR.iterdir():
        if old.is_file():
            old.unlink()

    # Pages is the browser-serving tier, so keep the entire synchronized
    # three-hour history same-origin. Assets are cropped to the much smaller
    # NEUS browser domain before they are written to site/history/.
    hot_window_minutes = max(
        30,
        int(os.environ.get("MRMS_PAGES_HOT_WINDOW_MINUTES", "180")),
    )
    hot_cutoff = now - timedelta(minutes=hot_window_minutes)

    files: dict[str, dict] = {}
    hot_frame_count = 0
    for frame in manifest.get("frames", []):
        frame_ts = frame_time(frame)
        if frame_ts is None or frame_ts < hot_cutoff:
            continue

        hot_frame_count += 1

        for key in ("radar_asset", "phase_asset", "precip_type_asset"):
            name = frame.get(key)
            if name and name in combined_assets:
                files[name] = combined_assets[name]

        # Manifest URLs are retained as a fallback, but Release asset URLs
        # from the combined partition index are preferred because they are
        # guaranteed to point at an existing asset.
        for key, api_key in (
            ("radar_asset", "radar_api_url"),
            ("phase_asset", "phase_api_url"),
            ("precip_type_asset", "precip_type_api_url"),
        ):
            name = frame.get(key)
            url = frame.get(api_key)
            if name and name not in files and url:
                files[name] = {"url": url}

    print(
        f"  Combined manifest: "
        f"{manifest.get('frame_count', len(manifest.get('frames', [])))} frames "
        f"across {len(manifest_candidates)} manifests"
    )
    print(
        f"  Pages hot buffer: {hot_frame_count} frames / "
        f"{hot_window_minutes} minutes"
    )
    if manifest.get("frames"):
        first_ts = manifest["frames"][0].get("timestamp_utc")
        last_ts = manifest["frames"][-1].get("timestamp_utc")
        print(f"  Combined history range: {first_ts} -> {last_ts}")
    print(f"  Release partitions searched: {len(release_assets)}")
    print(f"  Unique assets to stage: {len(files)}")

    if not files:
        print("  Manifest contains no stageable history assets yet.")
        return

    def worker(item: tuple[str, dict]) -> tuple[str, bool, str]:
        name, asset = item
        target = HISTORY_DIR / name
        last_error = ""
        # A Release asset can briefly return 404/5xx while GitHub finishes
        # updating the object. Retry before declaring it unavailable.
        for attempt in range(1, 4):
            try:
                data = download_asset(session, asset)
                if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
                    last_error = f"unexpected payload ({len(data):,} bytes)"
                else:
                    data = crop_history_asset_to_pages(data, tuple(
                        float(v) for v in (manifest.get("bounds") or DEFAULT_FULL_BOUNDS)
                    ))
                    target.write_bytes(data)
                    return name, True, f"{len(data):,} bytes (NEUS)"
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt < 3:
                import time
                time.sleep(1.5 * attempt)
        return name, False, last_error

    failures: list[tuple[str, str]] = []
    done = 0
    items = sorted(files.items())
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
        for name, ok, info in pool.map(worker, items):
            done += 1
            if not ok:
                failures.append((name, info))
            if done <= 5 or done == len(items) or done % 25 == 0:
                print(f"  [{done:>3}/{len(items):>3}] {'OK' if ok else 'FAIL'} {name} — {info}")

    print(f"  Staged: {done - len(failures)}")
    print(f"  Unavailable assets: {len(failures)}")

    # Never let one stale/missing Release blob take down the entire Pages
    # publication. Remove unavailable references from the manifest. Radar is
    # the required frame asset; if it is unavailable, drop only that frame.
    # Phase/precipitation are optional and can remain absent until a later run
    # repairs them from the persistent Release/archive store.
    unavailable = {name for name, _ in failures}
    if unavailable:
        sanitized_frames = []
        dropped_radar_frames = 0
        for frame in manifest.get("frames", []):
            radar_name = frame.get("radar_asset")
            if radar_name in unavailable:
                dropped_radar_frames += 1
                continue
            # A history frame is only useful to the composite viewer when
            # radar + phase + precipitation-type are all present. Drop the
            # entire frame on staging failure rather than publishing a blank
            # timestep with a null phase/ptype product.
            continue

        manifest["frames"] = sanitized_frames
        manifest["frame_count"] = len(sanitized_frames)
        manifest["phase_snapshot_count"] = sum(
            1 for frame in sanitized_frames if frame.get("phase_asset")
        )
        manifest["precip_type_snapshot_count"] = sum(
            1 for frame in sanitized_frames if frame.get("precip_type_asset")
        )
        manifest["staging_unavailable_assets"] = sorted(unavailable)
        manifest["staging_dropped_radar_frames"] = dropped_radar_frames
        manifest["staging_generated_at_utc"] = datetime.now(timezone.utc).isoformat()
        (OUTPUT / "mrms_history.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        for name, info in failures[:20]:
            print(f"    UNAVAILABLE {name}: {info}")
        print(
            f"  Pages publication will continue with {len(unavailable)} unavailable "
            f"asset(s); dropped radar frames: {dropped_radar_frames}"
        )

    print("PAGES HISTORY STAGING COMPLETE")


if __name__ == "__main__":
    main()
