from __future__ import annotations

"""Fetch the archive manifest/assets from GitHub Releases for Pages deployment."""

import concurrent.futures
import json
import os
from datetime import datetime, timezone
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
UA = "WinterRadar/1.1 (Pages history staging)"


def api_url(path: str) -> str:
    return f"{API}/repos/{REPO}{path}"


def h(accept: str = "application/vnd.github+json") -> dict[str, str]:
    return {
        "Accept": accept,
        "Authorization": f"Bearer {TOKEN}",
        "X-GitHub-Api-Version": API_VERSION,
        "User-Agent": UA,
    }


def get_release(session: requests.Session, tag: str) -> dict | None:
    r = session.get(api_url(f"/releases/tags/{quote(tag, safe='')}"), headers=h(), timeout=(20, 60))
    if r.status_code == 404:
        return None
    r.raise_for_status()
    data = r.json()
    data["assets"] = list_assets(session, int(data["id"]))
    return data


def list_assets(session: requests.Session, release_id: int) -> dict[str, dict]:
    assets: dict[str, dict] = {}
    page = 1
    while True:
        r = session.get(
            api_url(f"/releases/{release_id}/assets"),
            headers=h(),
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
        headers=h("application/octet-stream"),
        timeout=(20, 180),
    )
    r.raise_for_status()
    return r.content


def main() -> None:
    if not TOKEN or "/" not in REPO:
        raise SystemExit("GITHUB_TOKEN/GITHUB_REPOSITORY are required")

    now = datetime.now(timezone.utc)
    previous_day = now.date().toordinal() - 1
    tags = [f"mrms-{now:%Y%m%d}", f"mrms-{datetime.fromordinal(previous_day).date():%Y%m%d}"]

    session = requests.Session()
    release = None
    for tag in tags:
        release = get_release(session, tag)
        if release is not None:
            break
    if release is None:
        print("No MRMS history release exists yet; Pages will publish without history.")
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUT / "mrms_history.json").write_text(
            '{"version":"1.0-history","history_hours":3,"frame_count":0,"phase_snapshot_count":0,"precip_type_snapshot_count":0,"frames":[]}',
            encoding="utf-8",
        )
        return

    assets = release["assets"]
    manifest_asset = assets.get("mrms_history.json")
    if not manifest_asset:
        print(f"Release {release['tag']} has no mrms_history.json yet; Pages will publish without history.")
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        return

    manifest = json.loads(download_asset(session, manifest_asset).decode("utf-8"))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "mrms_history.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    status_asset = assets.get("mrms_archive_status.json")
    if status_asset:
        (OUTPUT / "mrms_archive_status.json").write_bytes(download_asset(session, status_asset))

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    for old in HISTORY_DIR.iterdir():
        if old.is_file():
            old.unlink()

    files: dict[str, str] = {}
    for frame in manifest.get("frames", []):
        for key, api_key in (("radar_asset", "radar_api_url"), ("phase_asset", "phase_api_url"), ("precip_type_asset", "precip_type_api_url")):
            name = frame.get(key)
            url = frame.get(api_key)
            if name and url:
                files[name] = url

    # Also tolerate manifest entries whose asset URL is stale but whose asset
    # is present in the current release list.
    for frame in manifest.get("frames", []):
        for key in ("radar_asset", "phase_asset", "precip_type_asset"):
            name = frame.get(key)
            if name and name in assets:
                files[name] = assets[name]["url"]

    print("=" * 68)
    print("WINTER RADAR — PAGES HISTORY STAGING")
    print("=" * 68)
    print(f"  Release: {release['tag']}")
    print(f"  Manifest frames: {manifest.get('frame_count', 0)}")
    print(f"  Unique assets to stage: {len(files)}")

    def worker(item: tuple[str, str]) -> tuple[str, bool, str]:
        name, url = item
        target = HISTORY_DIR / name
        try:
            data = download_asset(session, {"url": url})
            if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
                return name, False, f"unexpected payload ({len(data):,} bytes)"
            target.write_bytes(data)
            return name, True, f"{len(data):,} bytes"
        except Exception as exc:
            return name, False, f"{type(exc).__name__}: {exc}"

    failures = []
    done = 0
    items = sorted(files.items())
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
        for name, ok, info in pool.map(worker, items):
            done += 1
            print(f"  [{done:>3}/{len(items):>3}] {'OK' if ok else 'FAIL'} {name} — {info}")
            if not ok:
                failures.append((name, info))

    print(f"  Staged: {done - len(failures)}")
    print(f"  Failures: {len(failures)}")
    if failures:
        for name, info in failures[:10]:
            print(f"    FAILED {name}: {info}")
        raise SystemExit("History staging encountered download failures")

    print("PAGES HISTORY STAGING COMPLETE")


if __name__ == "__main__":
    main()
