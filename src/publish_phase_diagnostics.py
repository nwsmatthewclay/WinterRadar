#!/usr/bin/env python3
"""Publish the newest successful phase-evidence diagnostics to Release storage."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from archive_mrms_history import (
    GitHubReleaseStore,
    ensure_upload_release,
    history_release_date,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"
TOKEN = os.environ.get("GITHUB_TOKEN", "")
REPO = os.environ.get("GITHUB_REPOSITORY", "")

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


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def release_index_for_today(store: GitHubReleaseStore, today) -> dict:
    index = {today: []}
    for meta in store.list_history_releases():
        tag = str(meta.get("tag_name", ""))
        if history_release_date(tag) != today:
            continue
        release = store.get_release(tag)
        if release is not None:
            index[today].append(release)

    if not index[today]:
        index[today].append(store.ensure_release(f"mrms-{today:%Y%m%d}"))

    return index


def main() -> None:
    if not TOKEN or "/" not in REPO:
        raise SystemExit("GITHUB_TOKEN/GITHUB_REPOSITORY are required")

    metadata_path = OUTPUT / "mrms_current.json"
    if not metadata_path.exists():
        raise SystemExit("outputs/mrms_current.json is missing")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    mrms_time = metadata.get("mrms_time_utc")
    if not mrms_time:
        raise SystemExit("mrms_current.json has no mrms_time_utc")

    store = GitHubReleaseStore(TOKEN, REPO)
    now = utc_now()
    release_index = release_index_for_today(store, now.date())
    release = ensure_upload_release(store, now.date(), release_index)

    uploaded = []
    missing = []

    for name in BUNDLE:
        path = OUTPUT / name
        if not path.exists() or path.stat().st_size == 0:
            missing.append(name)
            continue

        content_type = "application/json" if name.endswith(".json") else "image/png"
        if name.endswith(".webp"):
            content_type = "image/webp"

        asset = store.upload_asset(
            release,
            name,
            path.read_bytes(),
            content_type,
            replace=True,
        )
        release.assets[asset["name"]] = asset
        uploaded.append(name)

    if missing:
        raise SystemExit(
            "Diagnostic bundle is incomplete; missing: " + ", ".join(missing)
        )

    status = {
        "source": "mrms-phase-diagnostics-workflow",
        "generated_at_utc": now.isoformat(),
        "mrms_time_utc": mrms_time,
        "release_tag": release.tag,
        "assets": uploaded,
    }

    store.upload_asset(
        release,
        "phase_diagnostics_latest.json",
        json.dumps(status, indent=2).encode("utf-8"),
        "application/json",
        replace=True,
    )

    print("=" * 68)
    print("PHASE DIAGNOSTICS PUBLISHED")
    print("=" * 68)
    print(f"  MRMS time: {mrms_time}")
    print(f"  Release: {release.tag}")
    print(f"  Assets: {len(uploaded)}")
    print("  Latest diagnostic bundle is ready for the live Pages workflow.")


if __name__ == "__main__":
    main()
