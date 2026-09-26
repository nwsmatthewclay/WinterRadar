from __future__ import annotations

"""
Download the MRMS 3-D vertical dual-polarization CAPPI products used by
WinterRadar.

Products:
  - MergedRhoHV
  - MergedZdr

Levels:
  0.50, 1.00, 1.50, 2.00, 2.50, 3.00, 3.50, 4.00 km

The MRMS 3-D archive is organized by product AND level:

  https://mrms.ncep.noaa.gov/3DRhoHV/MergedRhoHV_00.50/
  https://mrms.ncep.noaa.gov/3DZdr/MergedZdr_00.50/

The old downloader scanned the product root and therefore could not find
level-specific files. This version restores the proven level-directory layout
and adds a robust fallback when the MRMS `latest` gzip is temporarily truncated.
"""

import gzip
import html
import re
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from config import (
    DATA_DIR,
    MRMS_3D_RHOHV_BASE,
    MRMS_3D_ZDR_BASE,
    VERTICAL_DUALPOL_LEVELS_KM,
    VERTICAL_DUALPOL_PRODUCTS,
)


MAX_ATTEMPTS_LATEST = 3
MAX_ATTEMPTS_TIMESTAMPED = 3
REQUEST_TIMEOUT = (20, 120)

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": (
            "WinterRadar/1.3 "
            "(MRMS winter precipitation vertical dual-pol)"
        ),
        "Accept-Encoding": "identity",
        "Accept": "application/octet-stream,text/html;q=0.9,*/*;q=0.8",
    }
)

BASES = {
    "MergedRhoHV": MRMS_3D_RHOHV_BASE,
    "MergedZdr": MRMS_3D_ZDR_BASE,
}


def normalize_level(level) -> float:
    try:
        return float(level)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid MRMS vertical level: {level!r}") from exc


def level_string(level) -> str:
    """MRMS CAPPI directory/file level: 0.50 -> 00.50."""
    return f"{normalize_level(level):05.2f}"


def level_directory(product: str, level: float) -> str:
    return f"{BASES[product]}/{product}_{level_string(level)}"


def latest_url(product: str, level: float) -> str:
    level_text = level_string(level)
    return (
        f"{level_directory(product, level)}/"
        f"MRMS_{product}_{level_text}.latest.grib2.gz"
    )


def normalized_path(product: str, level: float) -> Path:
    level_text = level_string(level)
    directory = DATA_DIR / "dualpol" / f"{level_text}km"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"MRMS_{product}_{level_text}.latest.grib2"


def _atomic_download_and_decompress(url: str, destination: Path) -> tuple[int, int]:
    """
    Download a gzip payload and fully validate/decompress it.

    Returns:
        compressed_bytes, decompressed_bytes

    Nothing is promoted to `destination` until the complete gzip stream has
    been read successfully and the decompressed payload begins with GRIB.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        prefix="winterradar_dualpol_",
        suffix=".grib2.gz",
        dir=str(destination.parent),
        delete=False,
    ) as tmp:
        tmp_gz = Path(tmp.name)

    tmp_grib = destination.with_name(destination.name + ".part")

    compressed_bytes = 0
    decompressed_bytes = 0

    try:
        with SESSION.get(
            url,
            stream=True,
            timeout=REQUEST_TIMEOUT,
        ) as response:
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")
            print(
                f"    HTTP {response.status_code}; "
                f"content-type={content_type or 'unknown'}"
            )

            with tmp_gz.open("wb") as fh:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fh.write(chunk)
                        compressed_bytes += len(chunk)

        if compressed_bytes < 32:
            raise RuntimeError(
                f"downloaded gzip is unexpectedly small ({compressed_bytes:,} bytes)"
            )

        with gzip.open(tmp_gz, "rb") as src, tmp_grib.open("wb") as dst:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
                decompressed_bytes += len(chunk)

        if decompressed_bytes < 16:
            raise RuntimeError(
                f"decompressed GRIB is unexpectedly small ({decompressed_bytes:,} bytes)"
            )

        with tmp_grib.open("rb") as fh:
            if fh.read(4) != b"GRIB":
                raise RuntimeError("decompressed payload does not begin with GRIB")

        tmp_grib.replace(destination)
        return compressed_bytes, decompressed_bytes

    finally:
        tmp_gz.unlink(missing_ok=True)
        tmp_grib.unlink(missing_ok=True)


TIMESTAMPED_RE = re.compile(
    r"MRMS_(?P<product>[^_]+(?:RhoHV|Zdr))_"
    r"(?P<level>\d+\.\d{2})_"
    r"(?P<timestamp>\d{8}-\d{6})\.grib2\.gz"
)


def discover_timestamped_files(product: str, level: float) -> list[tuple[datetime, str]]:
    """
    Read the exact level directory and return timestamped files for that level,
    newest first.
    """
    directory = level_directory(product, level)
    print(f"    Scanning level directory: {directory}")

    response = SESSION.get(
        directory + "/",
        timeout=(20, 60),
    )
    response.raise_for_status()

    page = html.unescape(response.text)
    level_text = level_string(level)

    pattern = re.compile(
        rf"MRMS_{re.escape(product)}_{re.escape(level_text)}_"
        rf"(\d{{8}}-\d{{6}})\.grib2\.gz"
    )

    found: dict[datetime, str] = {}

    for match in pattern.finditer(page):
        timestamp = datetime.strptime(
            match.group(1),
            "%Y%m%d-%H%M%S",
        ).replace(tzinfo=timezone.utc)
        found[timestamp] = match.group(0)

    return sorted(
        found.items(),
        key=lambda item: item[0],
        reverse=True,
    )


def try_latest(product: str, level: float, destination: Path) -> bool:
    url = latest_url(product, level)

    print(f"  Latest URL:")
    print(f"    {url}")

    for attempt in range(1, MAX_ATTEMPTS_LATEST + 1):
        print(f"    Latest attempt {attempt}/{MAX_ATTEMPTS_LATEST}")

        try:
            compressed, decompressed = _atomic_download_and_decompress(
                url,
                destination,
            )
            print(
                f"    Gzip OK: {compressed:,} compressed bytes; "
                f"{decompressed:,} decompressed bytes"
            )
            print(f"    Final file: {destination}")
            return True

        except Exception as exc:
            print(f"    Latest attempt failed: {type(exc).__name__}: {exc}")
            destination.unlink(missing_ok=True)

            if attempt < MAX_ATTEMPTS_LATEST:
                delay = 2 ** (attempt - 1)
                print(f"    Retrying latest in {delay}s")
                time.sleep(delay)

    return False


def try_timestamped_fallback(
    product: str,
    level: float,
    destination: Path,
) -> bool:
    files = discover_timestamped_files(product, level)

    if not files:
        print("    No timestamped files found in the level directory.")
        return False

    print(f"    Timestamped candidates found: {len(files)}")

    # Try the newest few. A just-written MRMS file can also be incomplete, so
    # do not assume the first timestamped file is usable.
    for index, (timestamp, filename) in enumerate(
        files[:MAX_ATTEMPTS_TIMESTAMPED],
        start=1,
    ):
        url = f"{level_directory(product, level)}/{filename}"

        print(
            f"    Fallback {index}/{min(len(files), MAX_ATTEMPTS_TIMESTAMPED)}: "
            f"{filename}"
        )

        for attempt in range(1, MAX_ATTEMPTS_TIMESTAMPED + 1):
            try:
                compressed, decompressed = _atomic_download_and_decompress(
                    url,
                    destination,
                )

                print(
                    f"    Timestamped gzip OK: "
                    f"{compressed:,} compressed bytes; "
                    f"{decompressed:,} decompressed bytes"
                )
                print(
                    f"    Source time: "
                    f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC"
                )
                print(f"    Final file: {destination}")
                return True

            except Exception as exc:
                print(
                    f"    Timestamped attempt {attempt}/"
                    f"{MAX_ATTEMPTS_TIMESTAMPED} failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                destination.unlink(missing_ok=True)

                if attempt < MAX_ATTEMPTS_TIMESTAMPED:
                    time.sleep(1)

    return False


def download_level(product: str, level) -> Path:
    level = normalize_level(level)
    level_text = level_string(level)

    if product not in BASES:
        raise ValueError(f"No 3-D MRMS base URL configured for {product}")

    destination = normalized_path(product, level)

    print("")
    print("=" * 72)
    print(f"{product} @ {level_text} km")
    print("=" * 72)
    print(f"Directory: {level_directory(product, level)}")
    print(f"Destination: {destination}")

    # Primary path: official latest file.
    if try_latest(product, level, destination):
        return destination

    # Recovery path: newest complete timestamped observation.
    print("  Latest product was unavailable/incomplete.")
    print("  Switching to timestamped MRMS fallback.")

    if try_timestamped_fallback(product, level, destination):
        return destination

    raise RuntimeError(
        f"Unable to obtain a valid MRMS {product} file for "
        f"{level_text} km from either latest or timestamped sources."
    )


def main() -> None:
    print("")
    print("=" * 72)
    print("MRMS VERTICAL DUAL-POL DOWNLOAD")
    print("=" * 72)
    print(
        "Products: MergedRhoHV + MergedZdr | "
        "Levels: 0.50–4.00 km"
    )

    total = 0

    for _, product in VERTICAL_DUALPOL_PRODUCTS.items():
        for level in VERTICAL_DUALPOL_LEVELS_KM:
            download_level(product, level)
            total += 1

    print("")
    print("=" * 72)
    print(f"VERTICAL DOWNLOAD COMPLETE: {total} files")
    print("=" * 72)


if __name__ == "__main__":
    main()
