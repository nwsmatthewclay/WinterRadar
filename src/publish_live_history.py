from __future__ import annotations

"""Publish a buffered live radar/composite window into the persistent history store.

The live workflow already has the current MRMS scan and the current
phase/precipitation-type browser products. In addition to those current
products, this publisher places the newest buffered timestamped MRMS radar scans
into the GitHub Release history store.

Only the newest scan reuses the already-generated live WebP. The prior
scans are downloaded/rendered with the same authoritative
MergedReflectivityQCComposite source and full-CONUS Web Mercator geometry used
by the archive. This gives the Pages history viewer a near-live radar buffer
while the archive workflow catches up with the matching phase composites.

The archive no longer depends on the live watermark for eligibility, so these
frames are a freshness buffer rather than a correctness dependency.
"""

import json
import os
import shutil
import subprocess
import tempfile
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
UA = "WinterRadar/1.5 (buffered live radar/composite publisher)"

LIVE_RADAR_FRAMES = max(
    1,
    int(os.environ.get("MRMS_LIVE_HISTORY_FRAMES", "18")),
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
    radar_assets, phase_assets, precip_type_assets = gather_release_assets_many(all_history_releases)

    # Discover the timestamped MRMS directory directly. Do not wait for the
    # archive workflow or the live watermark to identify recent scans.
    observations = fetch_mrms_directory(store.session)
    observations = [
        obs for obs in observations
        if obs.valid_time < current_dt
    ]
    observations.sort(key=lambda item: item.valid_time)

    # The current live raster already exists locally. Always publish it first
    # so history cannot lose the newest scan merely because NOAA's timestamped
    # directory is a moment behind the latest product.
    current_release = ensure_upload_release(
        store,
        current_dt.date(),
        release_index,
    )

    current_radar_asset = f"radar_qc_conus_{current_stamp}.webp"
    current_radar_path = OUTPUT / "mrms_current_web.webp"
    if current_radar_asset not in radar_assets:
        if not current_radar_path.exists() or current_radar_path.stat().st_size == 0:
            raise SystemExit(
                f"Required live radar WebP is missing: {current_radar_path}"
            )
        data = current_radar_path.read_bytes()
        asset = store.upload_asset(
            current_release,
            current_radar_asset,
            data,
        )
        current_release.assets[asset["name"]] = asset
        radar_assets[asset["name"]] = asset
        print(
            f"  Stored current live radar {asset['name']} "
            f"({len(data):,} bytes) in {current_release.tag}"
        )
    else:
        print(f"  Already stored current live radar: {current_radar_asset}")

    # Add completed scans from a real time window strictly before the current
    # live scan. MRMS can publish irregular/extra observations, so a fixed
    # scan count can leave the first several minutes of history uncovered.
    prior_count = max(0, LIVE_RADAR_FRAMES - 1)
    buffer_window_minutes = max(
        20,
        int(os.environ.get("MRMS_LIVE_HISTORY_WINDOW_MINUTES", "30")),
    )
    buffer_cutoff = current_dt - timedelta(minutes=buffer_window_minutes)
    prior_candidates = [
        obs for obs in observations
        if buffer_cutoff <= obs.valid_time < current_dt
    ]
    prior_observations = prior_candidates[-prior_count:] if prior_count else []

    lats, lons = load_source_coordinates()
    bounds = read_history_bounds(OUTPUT / "mrms_current.json")
    y0, y1, x0, x1, crop_bounds = crop_indices(lats, lons, bounds)
    expected_shape = (int(lats.size), int(lons.size))
    row_map_cache: dict = {}
    phase_grib_paths: list[Path] = []
    phase_observations: list = []

    buffer_times = [obs.valid_time for obs in prior_observations] + [current_dt]

    print("=" * 68)
    print("WINTER RADAR — LIVE BUFFERED HISTORY HANDOFF")
    print("=" * 68)
    print(f"  Current live timestamp: {current_dt.isoformat()}")
    print(f"  Requested live radar buffer: {LIVE_RADAR_FRAMES} scans")
    print(
        "  Buffer range: "
        f"{min(buffer_times).isoformat()} -> {max(buffer_times).isoformat()}"
    )

    for obs in prior_observations:
        radar_missing = obs.asset_name not in radar_assets
        phase_name = f"phase_conus_{obs.valid_time.strftime('%Y%m%d-%H%M%S')}.webp"
        precip_name = f"preciptype_conus_{obs.valid_time.strftime('%Y%m%d-%H%M%S')}.webp"
        composites_missing = phase_name not in phase_assets or precip_name not in precip_type_assets

        if not radar_missing and not composites_missing:
            print(f"  Already complete: {obs.asset_name}")
            continue

        print(f"  Buffering historical radar/composite scan: {obs.valid_time.isoformat()}")
        data, grib_path = download_and_render_observation(
            store.session,
            obs,
            expected_shape,
            (y0, y1, x0, x1),
            crop_bounds,
            row_map_cache,
            keep_grib=composites_missing,
        )

        if radar_missing:
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

        if grib_path is not None:
            phase_grib_paths.append(grib_path)
            phase_observations.append(obs)

    # Build all matching phase and precipitation-type composites in ONE helper
    # process. build_history_precip_type.py caches RAP profiles by valid hour,
    # so an 18-scan live buffer normally requires only one RAP download per
    # model hour instead of one download per scan. A failure on one scan/hour
    # is isolated by the helper and can be retried by the next workflow run.
    if phase_grib_paths:
        helper_output = Path(tempfile.mkdtemp(prefix="winterradar_live_phase_"))
        try:
            helper = ROOT / "src" / "build_history_precip_type.py"
            print(
                f"  Building live-edge phase composites for {len(phase_grib_paths)} "
                "buffered scans with hourly RAP caching..."
            )
            cmd = [
                "python",
                str(helper),
                "--output-dir",
                str(helper_output),
                *[str(path) for path in phase_grib_paths],
            ]
            subprocess.run(cmd, check=True)

            for obs in phase_observations:
                names = [
                    f"phase_conus_{obs.valid_time.strftime('%Y%m%d-%H%M%S')}.webp",
                    f"preciptype_conus_{obs.valid_time.strftime('%Y%m%d-%H%M%S')}.webp",
                ]
                release = ensure_upload_release(
                    store,
                    obs.valid_time.date(),
                    release_index,
                )
                for name in names:
                    source = helper_output / name
                    if not source.exists():
                        print(f"  WARNING: live-edge composite deferred/missing: {name}")
                        continue
                    if name in release.assets:
                        continue
                    asset = store.upload_asset(
                        release,
                        name,
                        source.read_bytes(),
                        "image/webp",
                    )
                    release.assets[asset["name"]] = asset
                    print(
                        f"  Stored live-edge composite {name} "
                        f"({source.stat().st_size:,} bytes)"
                    )
        except Exception as exc:
            # Never let one RAP/MRMS problem prevent the current live products
            # from publishing. The archive workflow will retry the missing
            # composite on its next pass.
            print(
                "  WARNING: live-edge batch composite generation failed: "
                f"{type(exc).__name__}: {exc}"
            )
        finally:
            for path in phase_grib_paths:
                path.unlink(missing_ok=True)
            shutil.rmtree(helper_output, ignore_errors=True)

    # Always publish the current phase and per-scan precip-type products as the
    # exact live observation. The archive fills matching older phase scans
    # independently.
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
        "live_radar_buffer_count": len(buffer_times),
        "live_radar_buffer_oldest_utc": min(buffer_times).isoformat(),
        "live_radar_buffer_newest_utc": max(buffer_times).isoformat(),
        "source": "live-workflow",
    }

    store.upload_asset(
        current_release,
        "mrms_live_latest.json",
        json.dumps(watermark, indent=2).encode("utf-8"),
        "application/json",
        replace=True,
    )

    print("LIVE BUFFERED RADAR/COMPOSITE HISTORY HANDOFF COMPLETE")


if __name__ == "__main__":
    main()
