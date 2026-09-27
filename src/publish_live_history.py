from __future__ import annotations

"""Publish a fast five-scan live radar buffer into the persistent history store.

The live workflow already has the current MRMS scan and the current
phase/precipitation-type browser products. In addition to those current
products, this publisher places the newest five timestamped MRMS radar scans
into the GitHub Release history store.

Only the newest scan reuses the already-generated live WebP. The four prior
scans are downloaded/rendered with the same authoritative
MergedReflectivityQCComposite source and full-CONUS Web Mercator geometry used
by the archive. This gives the Pages history viewer a near-live radar buffer
while the archive workflow catches up with the matching phase composites.

The archive no longer depends on the live watermark for eligibility, so these
five frames are a freshness buffer rather than a correctness dependency.
"""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from archive_mrms_history import (
    GitHubReleaseStore,
    crop_indices,
    download_and_render_observation,
    ensure_upload_release,
    fetch_mrms_directory,
    gather_release_assets_many,
    load_source_coordinates,
    read_history_bounds,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"
TOKEN = os.environ.get("GITHUB_TOKEN", "")
REPO = os.environ.get("GITHUB_REPOSITORY", "")
UA = "WinterRadar/1.4 (five-scan live history publisher)"

LIVE_RADAR_FRAMES = max(
    1,
    int(os.environ.get("MRMS_LIVE_HISTORY_FRAMES", "5")),
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def active_release_index(store: GitHubReleaseStore, now: datetime) -> dict[object, list]:
    """Return all active history release partitions for today and yesterday."""
    today = now.date()
    yesterday = (now - timedelta(days=1)).date()
    dates = {today, yesterday}
    index = {d: [] for d in dates}

    for meta in store.list_history_releases():
        tag = str(meta.get("tag_name", ""))
        if not tag.startswith("mrms-"):
            continue

        date_text = tag[5:13]
        if len(date_text) != 8 or not date_text.isdigit():
            continue

        try:
            date_value = datetime.strptime(date_text, "%Y%m%d").date()
        except ValueError:
            continue

        if date_value not in dates:
            continue

        release = store.get_release(tag)
        if release is not None:
            index.setdefault(date_value, []).append(release)

    if not index[today]:
        index[today].append(store.ensure_release(f"mrms-{today:%Y%m%d}"))

    return index


def main() -> None:
    if not TOKEN or "/" not in REPO:
        raise SystemExit("GITHUB_TOKEN/GITHUB_REPOSITORY are required")

    meta_path = OUTPUT / "mrms_current.json"
    if not meta_path.exists():
        raise SystemExit("outputs/mrms_current.json is missing")

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    time_text = metadata.get("mrms_time_utc")
    if not time_text:
        raise SystemExit("mrms_current.json has no mrms_time_utc")

    current_dt = datetime.fromisoformat(
        str(time_text).replace("Z", "+00:00")
    ).astimezone(timezone.utc)
    current_stamp = current_dt.strftime("%Y%m%d-%H%M%S")

    store = GitHubReleaseStore(TOKEN, REPO)
    now = utc_now()
    release_index = active_release_index(store, now)

    all_history_releases = [
        rel
        for rels in release_index.values()
        for rel in rels
    ]
    radar_assets, _, _ = gather_release_assets_many(all_history_releases)

    # Discover the timestamped MRMS directory directly. Do not wait for the
    # archive workflow or the live watermark to identify the newest scans.
    observations = fetch_mrms_directory(store.session)
    observations = [
        obs
        for obs in observations
        if obs.valid_time <= now + timedelta(minutes=1)
    ]
    observations.sort(key=lambda item: item.valid_time)

    latest = observations[-LIVE_RADAR_FRAMES:]
    if not latest:
        raise RuntimeError("No recent timestamped MRMS observations were found")

    # Make sure the exact current live frame is represented even if the NOAA
    # timestamped directory temporarily trails the latest product.
    if current_dt not in {obs.valid_time for obs in latest}:
        matching = next(
            (obs for obs in observations if obs.valid_time == current_dt),
            None,
        )
        if matching is not None:
            latest.append(matching)
        latest = sorted(
            {obs.valid_time: obs for obs in latest}.values(),
            key=lambda item: item.valid_time,
        )[-LIVE_RADAR_FRAMES:]

    lats, lons = load_source_coordinates()
    bounds = read_history_bounds(OUTPUT / "mrms_current.json")
    y0, y1, x0, x1, crop_bounds = crop_indices(lats, lons, bounds)
    expected_shape = (int(lats.size), int(lons.size))
    row_map_cache: dict = {}

    print("=" * 68)
    print("WINTER RADAR — LIVE FIVE-SCAN HISTORY HANDOFF")
    print("=" * 68)
    print(f"  Current live timestamp: {current_dt.isoformat()}")
    print(f"  Requested live radar buffer: {LIVE_RADAR_FRAMES} scans")
    print(f"  NOAA newest selected scan: {latest[-1].valid_time.isoformat()}")
    print(
        "  Buffer range: "
        f"{latest[0].valid_time.isoformat()} -> {latest[-1].valid_time.isoformat()}"
    )

    for obs in latest:
        if obs.asset_name in radar_assets:
            print(f"  Already stored: {obs.asset_name}")
            continue

        if obs.valid_time == current_dt:
            live_path = OUTPUT / "mrms_current_web.webp"
            if live_path.exists() and live_path.stat().st_size > 0:
                data = live_path.read_bytes()
                print(
                    f"  Reusing live WebP for {obs.asset_name} "
                    f"({len(data):,} bytes)"
                )
            else:
                data, _ = download_and_render_observation(
                    store.session,
                    obs,
                    expected_shape,
                    (y0, y1, x0, x1),
                    crop_bounds,
                    row_map_cache,
                    keep_grib=False,
                )
        else:
            print(f"  Buffering historical radar scan: {obs.valid_time.isoformat()}")
            data, _ = download_and_render_observation(
                store.session,
                obs,
                expected_shape,
                (y0, y1, x0, x1),
                crop_bounds,
                row_map_cache,
                keep_grib=False,
            )

        release = ensure_upload_release(
            store,
            obs.valid_time.date(),
            release_index,
        )
        asset = store.upload_asset(release, obs.asset_name, data)
        release.assets[asset["name"]] = asset
        radar_assets[asset["name"]] = asset
        print(
            f"  Stored {asset['name']} ({len(data):,} bytes) "
            f"in {release.tag}"
        )

    # Always publish the current phase and per-scan precip-type products as the
    # exact live observation. The archive fills matching older phase scans
    # independently.
    current_release = ensure_upload_release(
        store,
        current_dt.date(),
        release_index,
    )

    current_files = {
        f"phase_conus_{current_stamp}.webp": (
            OUTPUT / "winter_phase_mask_web.webp",
            "image/webp",
        ),
        f"preciptype_conus_{current_stamp}.webp": (
            OUTPUT / "winter_precip_type_web.webp",
            "image/webp",
        ),
    }

    for name, (path, content_type) in current_files.items():
        if not path.exists() or path.stat().st_size == 0:
            raise SystemExit(f"Required live history output is missing: {path}")
        asset = store.upload_asset(
            current_release,
            name,
            path.read_bytes(),
            content_type,
        )
        current_release.assets[asset["name"]] = asset

    watermark = {
        "generated_at_utc": utc_now().isoformat(),
        "mrms_time_utc": current_dt.isoformat(),
        "release_tag": current_release.tag,
        "radar_asset": f"radar_qc_conus_{current_stamp}.webp",
        "phase_asset": f"phase_conus_{current_stamp}.webp",
        "precip_type_asset": f"preciptype_conus_{current_stamp}.webp",
        "live_radar_buffer_count": len(latest),
        "live_radar_buffer_oldest_utc": latest[0].valid_time.isoformat(),
        "live_radar_buffer_newest_utc": latest[-1].valid_time.isoformat(),
        "source": "live-workflow",
    }

    store.upload_asset(
        current_release,
        "mrms_live_latest.json",
        json.dumps(watermark, indent=2).encode("utf-8"),
        "application/json",
        replace=True,
    )

    print("LIVE FIVE-SCAN HISTORY HANDOFF COMPLETE")


if __name__ == "__main__":
    main()
