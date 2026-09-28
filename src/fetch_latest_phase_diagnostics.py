#!/usr/bin/env python3
"""Restore the latest successful phase-evidence diagnostic bundle."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"
TOKEN = os.environ.get("GITHUB_TOKEN", "")
REPO = os.environ.get("GITHUB_REPOSITORY", "")
API = "https://api.github.com"
API_VERSION = "2026-03-10"
UA = "WinterRadar/1.6 (phase diagnostics restore)"

BUNDLE = (
    "phase_radar_fusion.png",
    "phase_radar_fusion_overlay.png",
    "phase_radar_fusion_overlay_web.png",
    "phase_radar_fusion_overlay_web.webp",
    "phase_radar_fusion.json",
    "phase_agreement.png",
    "phase_agreement_overlay.png",
    "phase_agreement_overlay_web.png",
    "phase_agreement_overlay_web.webp",
    "phase_agreement.json",
    "phase_agreement_click.json",
)

TAG_RE = re.compile(r"^mrms-(\d{8})(?:-\d+)?$")


def headers(accept: str = "application/vnd.github+json") -> dict[str, str]:
    return {
        "Accept": accept,
        "Authorization": f"Bearer {TOKEN}",
        "X-GitHub-Api-Version": API_VERSION,
        "User-Agent": UA,
    }


def api_url(path: str) -> str:
    return f"{API}/repos/{REPO}{path}"


def list_releases(session: requests.Session) -> list[dict]:
    out = []
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
        out.extend(batch)
        if len(batch) < 100:
            return out
        page += 1


def list_assets(session: requests.Session, release_id: int) -> dict[str, dict]:
    out = {}
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
        for asset in batch:
            out[str(asset["name"])] = asset
        if len(batch) < 100:
            return out
        page += 1


def release_sort_key(release: dict) -> tuple[str, int]:
    tag = str(release.get("tag_name", ""))
    suffix = 0
    if "-" in tag:
        try:
            suffix = int(tag.rsplit("-", 1)[1])
        except ValueError:
            suffix = 0
    return (str(release.get("updated_at", "")), suffix)


def download_asset(session: requests.Session, asset: dict) -> bytes:
    r = session.get(
        asset["url"],
        headers=headers("application/octet-stream"),
        timeout=(20, 180),
    )
    r.raise_for_status()
    return r.content


def main() -> None:
    if not TOKEN or "/" not in REPO:
        raise SystemExit("GITHUB_TOKEN/GITHUB_REPOSITORY are required")

    session = requests.Session()
    now = datetime.now(timezone.utc)
    wanted_dates = {now.date(), (now - timedelta(days=1)).date()}

    candidates = []
    for release in list_releases(session):
        match = TAG_RE.match(str(release.get("tag_name", "")))
        if not match:
            continue
        try:
            date_value = datetime.strptime(match.group(1), "%Y%m%d").date()
        except ValueError:
            continue
        if date_value in wanted_dates:
            candidates.append(release)

    candidates.sort(key=release_sort_key, reverse=True)

    chosen = None
    chosen_assets = None
    chosen_status = None

    for release in candidates:
        assets = list_assets(session, int(release["id"]))
        status_asset = assets.get("phase_diagnostics_latest.json")
        if not status_asset or not all(name in assets for name in BUNDLE):
            continue

        try:
            status = json.loads(
                download_asset(session, status_asset).decode("utf-8")
            )
        except Exception as exc:
            print(
                f"  WARNING: unable to read diagnostic bundle in "
                f"{release.get('tag_name')}: {type(exc).__name__}: {exc}"
            )
            continue

        chosen = release
        chosen_assets = assets
        chosen_status = status
        break

    if chosen is None:
        print(
            "  No successful phase diagnostic bundle is available yet; "
            "keeping fallback files."
        )
        return

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name in BUNDLE:
        data = download_asset(session, chosen_assets[name])
        if len(data) < 16:
            raise RuntimeError(
                f"Diagnostic asset {name} is unexpectedly small ({len(data)} bytes)"
            )
        (OUTPUT / name).write_bytes(data)

    (OUTPUT / "phase_diagnostics_latest.json").write_text(
        json.dumps(chosen_status, indent=2),
        encoding="utf-8",
    )

    print(
        "  Restored phase diagnostic bundle from "
        f"{chosen.get('tag_name')} for MRMS time "
        f"{chosen_status.get('mrms_time_utc', 'unknown')}"
    )


if __name__ == "__main__":
    main()
