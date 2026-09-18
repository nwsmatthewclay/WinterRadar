from __future__ import annotations

import html
import re
from datetime import datetime
from pathlib import Path

import requests

from config import (
    DATA_DIR,
    MRMS_BASE,
    VERTICAL_DUALPOL_LEVELS_KM,
    VERTICAL_DUALPOL_PRODUCTS,
)
from download_mrms import download_gzip


SESSION = requests.Session()

SESSION.headers.update(
    {
        "User-Agent": (
            "WinterRadar/1.0 "
            "(MRMS winter precipitation visualization project)"
        ),
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Encoding": "identity",
    }
)


def discover_files(product: str) -> list[tuple[float, datetime, str]]:
    """
    Find timestamped MRMS 3-D files for a product.

    Example:
        MRMS_MergedRhoHV_00.50_20260918-184541.grib2.gz
    """

    url = f"{MRMS_BASE}/{product}/"

    print("")
    print(f"Scanning MRMS directory:")
    print(f"  {url}")

    response = SESSION.get(
        url,
        timeout=(20, 60),
    )

    response.raise_for_status()

    page = html.unescape(response.text)

    # IMPORTANT:
    # Require the full MRMS product name so we don't accidentally
    # capture only "_00.50_..." from the middle of the filename.
    pattern = re.compile(
        rf"MRMS_{re.escape(product)}_"
        rf"(\d+\.\d{{2}})_"
        rf"(\d{{8}}-\d{{6}})\.grib2\.gz"
    )

    results = []

    for match in pattern.finditer(page):

        level_km = float(match.group(1))

        timestamp = datetime.strptime(
            match.group(2),
            "%Y%m%d-%H%M%S",
        )

        filename = match.group(0)

        results.append(
            (
                level_km,
                timestamp,
                filename,
            )
        )

    # Remove duplicates while preserving the newest instance.
    unique = {}

    for level, timestamp, filename in results:
        key = (level, timestamp)

        unique[key] = (
            level,
            timestamp,
            filename,
        )

    return list(unique.values())


def find_latest_level_file(
    product: str,
    target_level: float,
):
    """
    Find the newest available file for a requested
    vertical level.
    """

    files = discover_files(product)

    matches = [
        item
        for item in files
        if abs(item[0] - target_level) < 0.001
    ]

    if not matches:
        raise RuntimeError(
            f"No MRMS {product} file found for "
            f"{target_level:.2f} km"
        )

    matches.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return matches[0]


def normalized_path(
    product: str,
    level_km: float,
) -> Path:

    directory = (
        DATA_DIR
        / "dualpol"
        / f"{level_km:.2f}km"
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        directory
        / (
            f"MRMS_{product}_"
            f"{level_km:.2f}.latest.grib2"
        )
    )


def write_timestamp(
    product: str,
    level_km: float,
    timestamp: datetime,
) -> None:

    directory = (
        DATA_DIR
        / "dualpol"
        / f"{level_km:.2f}km"
    )

    timestamp_path = (
        directory
        / f"{product}.timestamp.txt"
    )

    timestamp_path.write_text(
        timestamp.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        encoding="utf-8",
    )


def download_level(
    product: str,
    level_km: float,
) -> None:

    print("")
    print(
        f"{product} @ {level_km:.2f} km"
    )
    print("-" * 60)

    (
        found_level,
        timestamp,
        filename,
    ) = find_latest_level_file(
        product,
        level_km,
    )

    url = (
        f"{MRMS_BASE}/{product}/{filename}"
    )

    destination = normalized_path(
        product,
        found_level,
    )

    print(
        f"Source level: {found_level:.2f} km"
    )

    print(
        f"Source time:  "
        f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC"
    )

    print(
        f"Source file:  {filename}"
    )

    print(
        f"Destination:  {destination}"
    )

    download_gzip(
        url,
        destination,
    )

    write_timestamp(
        product,
        found_level,
        timestamp,
    )


def main() -> None:

    print("")
    print("=" * 72)
    print("MRMS VERTICAL DUAL-POL DOWNLOAD")
    print("=" * 72)

    total = 0

    for _, product in VERTICAL_DUALPOL_PRODUCTS.items():

        for level_km in VERTICAL_DUALPOL_LEVELS_KM:

            download_level(
                product,
                level_km,
            )

            total += 1

    print("")
    print("=" * 72)
    print(
        f"VERTICAL DOWNLOAD COMPLETE: "
        f"{total} files"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
