from __future__ import annotations

"""Publish the live WinterRadar outputs into the persistent MRMS history release.

The live workflow already has the newest MRMS scan plus the current phase and
precipitation-type browser rasters. This script stores those exact files under
the same timestamped names used by the history collector, then writes a small
watermark file. The archive workflow uses the watermark as its upper boundary,
so it never re-downloads a scan that the live workflow has already completed.
"""

import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"
TOKEN = os.environ.get("GITHUB_TOKEN", "")
REPO = os.environ.get("GITHUB_REPOSITORY", "")
API = "https://api.github.com"
API_VERSION = "2026-03-10"
UA = "WinterRadar/1.1 (live history publisher)"
TIMEOUT = (20, 120)
UPLOAD_TIMEOUT = (20, 180)


def api_url(path: str) -> str:
    return f"{API}/repos/{REPO}{path}"


def headers(accept: str = "application/vnd.github+json") -> dict[str, str]:
    return {
        "Accept": accept,
        "Authorization": f"Bearer {TOKEN}",
        "X-GitHub-Api-Version": API_VERSION,
        "User-Agent": UA,
    }


def request(session: requests.Session, method: str, path: str, **kwargs) -> requests.Response:
    r = session.request(method, api_url(path), timeout=TIMEOUT, **kwargs)
    if not r.ok:
        raise RuntimeError(f"GitHub API {method} {path} failed: HTTP {r.status_code}: {r.text[:500]}")
    return r


def get_release(session: requests.Session, tag: str) -> dict | None:
    r = session.get(api_url(f"/releases/tags/{quote(tag, safe='')}"), headers=headers(), timeout=TIMEOUT)
    if r.status_code == 404:
        return None
    if not r.ok:
        raise RuntimeError(f"Release lookup failed: HTTP {r.status_code}: {r.text[:500]}")
    data = r.json()
    data["assets"] = list_assets(session, int(data["id"]))
    return data


def list_assets(session: requests.Session, release_id: int) -> dict[str, dict]:
    out: dict[str, dict] = {}
    page = 1
    while True:
        r = session.get(
            api_url(f"/releases/{release_id}/assets"),
            headers=headers(),
            params={"per_page": 100, "page": page},
            timeout=TIMEOUT,
        )
        if not r.ok:
            raise RuntimeError(f"Asset listing failed: HTTP {r.status_code}: {r.text[:500]}")
        batch = r.json()
        for item in batch:
            out[str(item["name"])] = item
        if len(batch) < 100:
            return out
        page += 1


def create_release(session: requests.Session, tag: str) -> dict:
    payload = {
        "tag_name": tag,
        "name": f"WinterRadar MRMS 3-Hour CONUS History — {tag.removeprefix('mrms-')}",
        "body": "Automated rolling 3-hour WinterRadar MRMS observation archive.",
        "draft": False,
        "prerelease": False,
        "generate_release_notes": False,
    }
    r = request(session, "POST", "/releases", json=payload)
    data = r.json()
    data["assets"] = {}
    return data


def ensure_release(session: requests.Session, tag: str) -> dict:
    return get_release(session, tag) or create_release(session, tag)


def delete_asset(session: requests.Session, asset: dict) -> None:
    r = session.delete(
        api_url(f"/releases/assets/{int(asset['id'])}"),
        headers=headers(),
        timeout=TIMEOUT,
    )
    if r.status_code not in (204, 404):
        raise RuntimeError(f"Delete failed for {asset.get('name')}: HTTP {r.status_code}: {r.text[:500]}")


def upload(session: requests.Session, release: dict, name: str, data: bytes, content_type: str, replace: bool = False) -> dict:
    assets = release.setdefault("assets", {})
    existing = assets.get(name)
    if existing and not replace:
        print(f"  Already stored: {name}")
        return existing
    if existing and replace:
        delete_asset(session, existing)
        assets.pop(name, None)
        print(f"  Replacing: {name}")
    upload_url = str(release["upload_url"]).split("{", 1)[0]
    r = session.post(
        upload_url,
        params={"name": name},
        headers={**headers(), "Content-Type": content_type},
        data=data,
        timeout=UPLOAD_TIMEOUT,
    )
    if r.status_code == 422:
        refreshed = list_assets(session, int(release["id"]))
        if name in refreshed:
            assets[name] = refreshed[name]
            print(f"  Upload already completed: {name}")
            return refreshed[name]
    if not r.ok:
        raise RuntimeError(f"Upload failed for {name}: HTTP {r.status_code}: {r.text[:500]}")
    item = r.json()
    assets[name] = item
    print(f"  Stored {name} ({len(data):,} bytes)")
    return item


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
    dt = datetime.fromisoformat(str(time_text).replace("Z", "+00:00")).astimezone(timezone.utc)
    stamp = dt.strftime("%Y%m%d-%H%M%S")
    tag = f"mrms-{dt:%Y%m%d}"

    files = {
        f"radar_qc_conus_{stamp}.webp": (OUTPUT / "mrms_current_web.webp", "image/webp"),
        f"phase_conus_{stamp}.webp": (OUTPUT / "winter_phase_mask_web.webp", "image/webp"),
        f"preciptype_conus_{stamp}.webp": (OUTPUT / "winter_precip_type_web.webp", "image/webp"),
    }
    for name, (path, _ctype) in files.items():
        if not path.exists() or path.stat().st_size == 0:
            raise SystemExit(f"Required live history output is missing: {path}")

    session = requests.Session()
    release = ensure_release(session, tag)

    print("=" * 68)
    print("WINTER RADAR — LIVE HISTORY HANDOFF")
    print("=" * 68)
    print(f"  Live MRMS timestamp: {dt.isoformat()}")
    print(f"  Release: {release['tag']}")

    for name, (path, ctype) in files.items():
        upload(session, release, name, path.read_bytes(), ctype)

    watermark = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "mrms_time_utc": dt.isoformat(),
        "release_tag": tag,
        "radar_asset": f"radar_qc_conus_{stamp}.webp",
        "phase_asset": f"phase_conus_{stamp}.webp",
        "precip_type_asset": f"preciptype_conus_{stamp}.webp",
        "source": "live-workflow",
    }
    watermark_bytes = json.dumps(watermark, indent=2).encode("utf-8")
    upload(session, release, "mrms_live_latest.json", watermark_bytes, "application/json", replace=True)

    print("LIVE HISTORY HANDOFF COMPLETE")


if __name__ == "__main__":
    main()
