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

LEVEL_PATTERN = re.compile(
    r"_(\d+\.\d{2})_(\d{8}-\d{6})\.grib2\.gz"
)


def discover_files(product: str) -> list[tuple[float, datetime, str]]:
    """
    Find recent historical MRMS files for a product.

    Returns:
        (level_km, timestamp, filename)
    """

    url = f"{MRMS_BASE}/{product}/?C=M;O=D"

    print(f"Scanning MRMS directory:")
    print(f"  {url}")

    response = SESSION.get(
        url,
        timeout=(20, 60),
    )

    response.raise_for_status()

    page = html.unescape(response.text)

    results = []

    for match in LEVEL_PATTERN.finditer(page):

        level = float(match.group(1))
        timestamp_string = match.group(2)

        timestamp = datetime.strptime(
            timestamp_string,
            "%Y%m%d-%H%M%S",
        )

        filename = match.group(0)

        results.append(
            (
                level,
                timestamp,
                filename,
            )
        )

    return results


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
            f"No {product} file found for "
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
    """Return our stable local filename."""

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
        / f"MRMS_{product}_{level_km:.2f}.latest.grib2"
    )


def write_timestamp(
    product: str,
    level_km: float,
    timestamp: datetime,
):
    """Record the source timestamp."""

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
        timestamp.isoformat(),
        encoding="utf-8",
    )


def download_level(
    product: str,
    level_km: float,
):
    print("")
    print(
        f"{product} @ {level_km:.2f} km"
    )
    print("-" * 60)

    _, timestamp, filename = find_latest_level_file(
        product,
        level_km,
    )

    url = (
        f"{MRMS_BASE}/{product}/{filename}"
    )

    destination = normalized_path(
        product,
        level_km,
    )

    print(f"Source time: {timestamp} UTC")
    print(f"Source file: {filename}")
    print(f"Destination: {destination}")

    download_gzip(
        url,
        destination,
    )

    write_timestamp(
        product,
        level_km,
        timestamp,
    )


def main():
    print("")
    print("=" * 72)
    print("MRMS VERTICAL DUAL-POL DOWNLOAD")
    print("=" * 72)

    total = 0

    for name, product in VERTICAL_DUALPOL_PRODUCTS.items():

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
