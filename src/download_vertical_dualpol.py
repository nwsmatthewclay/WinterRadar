from __future__ import annotations

import gzip
import html
import re
import shutil
from datetime import datetime
from pathlib import Path

import requests

from config import (
    DATA_DIR,
    MRMS_3D_RHOHV_BASE,
    MRMS_3D_ZDR_BASE,
    VERTICAL_DUALPOL_LEVELS_KM,
    VERTICAL_DUALPOL_PRODUCTS,
)

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "WinterRadar/1.1 (MRMS winter precipitation visualization project)",
    "Accept-Encoding": "identity",
})

BASES = {
    "MergedRhoHV": MRMS_3D_RHOHV_BASE,
    "MergedZdr": MRMS_3D_ZDR_BASE,
}


def discover_files(product: str) -> list[tuple[float, datetime, str]]:
    base = BASES[product]
    response = SESSION.get(f"{base}/", timeout=(20, 60))
    response.raise_for_status()
    page = html.unescape(response.text)
    pattern = re.compile(
        rf"MRMS_{re.escape(product)}_(\d+\.\d{{2}})_(\d{{8}}-\d{{6}})\.grib2\.gz"
    )
    results = []
    for match in pattern.finditer(page):
        results.append((
            float(match.group(1)),
            datetime.strptime(match.group(2), "%Y%m%d-%H%M%S"),
            match.group(0),
        ))
    return list({(a, b): (a, b, c) for a, b, c in results}.values())


def find_latest_level_file(product: str, target_level: float):
    matches = [x for x in discover_files(product) if abs(x[0] - target_level) < 0.001]
    if not matches:
        raise RuntimeError(f"No MRMS {product} file found for {target_level:.2f} km")
    return max(matches, key=lambda x: x[1])


def normalized_path(product: str, level_km: float) -> Path:
    directory = DATA_DIR / "dualpol" / f"{level_km:.2f}km"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"MRMS_{product}_{level_km:.2f}.latest.grib2"


def download_gzip(url: str, destination: Path) -> None:
    tmp_gz = destination.with_suffix(destination.suffix + ".gz.part")
    tmp_grib = destination.with_suffix(destination.suffix + ".part")
    with SESSION.get(url, stream=True, timeout=(20, 120)) as response:
        response.raise_for_status()
        with tmp_gz.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
    try:
        with gzip.open(tmp_gz, "rb") as src, tmp_grib.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
        tmp_grib.replace(destination)
    finally:
        tmp_gz.unlink(missing_ok=True)
        tmp_grib.unlink(missing_ok=True)


def download_level(product: str, level_km: float) -> None:
    found_level, timestamp, filename = find_latest_level_file(product, level_km)
    base = BASES[product]
    url = f"{base}/{filename}"
    destination = normalized_path(product, found_level)
    print(f"{product} @ {found_level:.2f} km — {timestamp:%Y-%m-%d %H:%M:%S} UTC")
    print(f"  {url}")
    download_gzip(url, destination)
    (destination.parent / f"{product}.timestamp.txt").write_text(
        timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"), encoding="utf-8"
    )


def main() -> None:
    print("=" * 72)
    print("MRMS VERTICAL DUAL-POL DOWNLOAD")
    print("=" * 72)
    total = 0
    for product in VERTICAL_DUALPOL_PRODUCTS.values():
        for level in VERTICAL_DUALPOL_LEVELS_KM:
            download_level(product, level)
            total += 1
    print(f"VERTICAL DOWNLOAD COMPLETE: {total} files")


if __name__ == "__main__":
    main()
