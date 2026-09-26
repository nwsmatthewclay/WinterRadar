#!/usr/bin/env python3
"""Stage persistent MRMS Release history into the GitHub Pages tree.

GitHub Actions runners are ephemeral, so historical MRMS files live in GitHub
Release assets.  This script reads the history manifest from the Release
partitions, downloads the referenced WebP assets, and places them in
site_history_tmp for the Pages build.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"
HISTORY_DIR = ROOT / "site_history_tmp"
TOKEN = os.environ.get("GITHUB_TOKEN", "")
REPO = os.environ.get("GITHUB_REPOSITORY", "")
API = "https://api.github.com"
API_VERSION = "2026-03-10"
UA = "WinterRadar/1.2 (Pages history staging)"


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

    # Prefer the manifest advertising the greatest number of frames. If two
    # partitions somehow contain the same count, prefer the most recently
    # updated release.
    manifest_candidates.sort(
        key=lambda item: (
            int(item[2].get("frame_count", len(item[2].get("frames", [])))),
            str(item[0].get("updated_at", "")),
        ),
        reverse=True,
    )
    release, _, manifest = manifest_candidates[0]

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

    files: dict[str, dict] = {}
    for frame in manifest.get("frames", []):
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

    print(f"  Selected manifest release: {release.get('tag_name', 'unknown')}")
    print(f"  Manifest frames: {manifest.get('frame_count', len(manifest.get('frames', [])))}")
    print(f"  Release partitions searched: {len(release_assets)}")
    print(f"  Unique assets to stage: {len(files)}")

    if not files:
        print("  Manifest contains no stageable history assets yet.")
        return

    def worker(item: tuple[str, dict]) -> tuple[str, bool, str]:
        name, asset = item
        target = HISTORY_DIR / name
        try:
            data = download_asset(session, asset)
            if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
                return name, False, f"unexpected payload ({len(data):,} bytes)"
            target.write_bytes(data)
            return name, True, f"{len(data):,} bytes"
        except Exception as exc:
            return name, False, f"{type(exc).__name__}: {exc}"

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
    print(f"  Failures: {len(failures)}")
    if failures:
        for name, info in failures[:10]:
            print(f"    FAILED {name}: {info}")
        raise SystemExit("History staging encountered download failures")

    print("PAGES HISTORY STAGING COMPLETE")


if __name__ == "__main__":
    main()
