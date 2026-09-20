from __future__ import annotations

import gzip
import os
import shutil
import time
from pathlib import Path

import requests

from config import (
    DATA_DIR,
    MRMS_3D_RHOHV_BASE,
    MRMS_3D_ZDR_BASE,
    VERTICAL_DUALPOL_LEVELS_KM,
    VERTICAL_DUALPOL_PRODUCTS,
)


# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------

MAX_ATTEMPTS = 5
CONNECT_TIMEOUT = 20
READ_TIMEOUT = 180
CHUNK_SIZE = 1024 * 1024
RETRYABLE_STATUS = (408, 429, 500, 502, 503, 504)

USER_AGENT = "WinterRadar/1.1 (MRMS vertical dual-pol downloader)"


# ----------------------------------------------------------------------
# MRMS 3-D product locations
# ----------------------------------------------------------------------

BASE_URLS = {
    "MergedRhoHV": MRMS_3D_RHOHV_BASE,
    "MergedZdr": MRMS_3D_ZDR_BASE,
}


# ----------------------------------------------------------------------
# HTTP helpers
# ----------------------------------------------------------------------

def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "*/*",
        }
    )
    return session


def _validate_grib(path: Path) -> None:
    if not path.exists() or path.stat().st_size < 16:
        raise RuntimeError(f"Downloaded GRIB is missing or too small: {path}")

    with path.open("rb") as fh:
        magic = fh.read(4)

    if magic != b"GRIB":
        raise RuntimeError(
            f"Downloaded file is not a GRIB2 payload "
            f"(starts with {magic!r}): {path}"
        )


def _validate_gzip_grib(path: Path) -> int:
    """Fully consume and validate a gzip-wrapped GRIB file."""
    if not path.exists() or path.stat().st_size < 16:
        raise RuntimeError(
            f"Downloaded gzip payload is missing or too small: {path}"
        )

    total = 0

    try:
        with gzip.open(path, "rb") as fh:
            magic = fh.read(4)

            if magic != b"GRIB":
                raise RuntimeError(
                    "Compressed response does not contain a GRIB payload: "
                    f"{path}"
                )

            while True:
                chunk = fh.read(CHUNK_SIZE)
                if not chunk:
                    break
                total += len(chunk)

    except (EOFError, OSError, gzip.BadGzipFile) as exc:
        raise RuntimeError(
            f"Incomplete/corrupt gzip stream: {path}: {exc}"
        ) from exc

    if total < 16:
        raise RuntimeError(
            f"Decompressed GRIB payload is unexpectedly small: {total} bytes"
        )

    return total


def download_gzip(
    url: str,
    destination: Path,
) -> Path:
    """
    Download a gzip-wrapped GRIB and atomically decompress it.

    This replaces the removed download_mrms.download_gzip() helper so the
    vertical dual-pol downloader is independent of the live 2-D downloader.
    """

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    gz_path = destination.with_suffix(
        destination.suffix + ".download"
    )
    grib_tmp = destination.with_suffix(
        destination.suffix + ".tmp"
    )

    last_error: Exception | None = None

    with _session() as session:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            print(
                f"  Download attempt "
                f"{attempt}/{MAX_ATTEMPTS}"
            )

            try:
                gz_path.unlink(missing_ok=True)
                grib_tmp.unlink(missing_ok=True)
                destination.unlink(missing_ok=True)

                with session.get(
                    url,
                    timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
                    stream=True,
                    allow_redirects=True,
                ) as response:

                    if response.status_code in RETRYABLE_STATUS:
                        raise requests.HTTPError(
                            f"Transient HTTP {response.status_code}",
                            response=response,
                        )

                    response.raise_for_status()

                    expected = response.headers.get("Content-Length")
                    expected_bytes = (
                        int(expected)
                        if expected and expected.isdigit()
                        else None
                    )

                    received = 0

                    with gz_path.open("wb") as fh:
                        for chunk in response.iter_content(
                            chunk_size=CHUNK_SIZE
                        ):
                            if chunk:
                                fh.write(chunk)
                                received += len(chunk)

                if received < 16:
                    raise RuntimeError(
                        f"Downloaded payload is unexpectedly small: "
                        f"{received} bytes"
                    )

                if (
                    expected_bytes is not None
                    and received != expected_bytes
                ):
                    raise RuntimeError(
                        f"Incomplete HTTP body: received "
                        f"{received:,} of {expected_bytes:,} bytes"
                    )

                decompressed_bytes = _validate_gzip_grib(gz_path)

                print(
                    f"  Gzip OK: "
                    f"{received:,} compressed bytes; "
                    f"{decompressed_bytes:,} decompressed bytes"
                )

                with gzip.open(gz_path, "rb") as src, grib_tmp.open(
                    "wb"
                ) as dst:
                    while True:
                        chunk = src.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        dst.write(chunk)

                _validate_grib(grib_tmp)
                os.replace(grib_tmp, destination)

                gz_path.unlink(missing_ok=True)

                print(
                    f"  Final file: {destination}"
                )

                return destination

            except Exception as exc:
                last_error = exc

                gz_path.unlink(missing_ok=True)
                grib_tmp.unlink(missing_ok=True)
                destination.unlink(missing_ok=True)

                print(
                    f"  Download failed: "
                    f"{type(exc).__name__}: {exc}"
                )

                if attempt < MAX_ATTEMPTS:
                    wait_seconds = 5 * attempt
                    print(
                        f"  Retrying in "
                        f"{wait_seconds} seconds..."
                    )
                    time.sleep(wait_seconds)

    raise RuntimeError(
        f"Unable to download {url} after "
        f"{MAX_ATTEMPTS} attempts. "
        f"Last error: {last_error}"
    )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def normalize_level(level) -> float:
    try:
        return float(level)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid MRMS vertical level: {level!r}"
        ) from exc


def level_string(level) -> str:
    level = normalize_level(level)
    return f"{level:05.2f}"


def local_path(product: str, level) -> Path:
    level = normalize_level(level)
    level_text = level_string(level)

    directory = (
        DATA_DIR
        / "dualpol"
        / f"{level_text}km"
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        directory
        / f"MRMS_{product}_{level_text}.latest.grib2"
    )


# ----------------------------------------------------------------------
# Download one CAPPI
# ----------------------------------------------------------------------

def download_level(
    product: str,
    level,
) -> Path:

    level = normalize_level(level)
    level_text = level_string(level)

    print("")
    print("=" * 72)
    print(f"{product} @ {level_text} km")
    print("=" * 72)

    if product not in BASE_URLS:
        raise ValueError(
            f"No 3-D MRMS base URL configured for {product}"
        )

    base_url = BASE_URLS[product]

    directory_url = (
        f"{base_url}/"
        f"{product}_{level_text}"
    )

    filename = (
        f"MRMS_{product}_"
        f"{level_text}.latest.grib2.gz"
    )

    url = (
        f"{directory_url}/"
        f"{filename}"
    )

    destination = local_path(
        product,
        level,
    )

    print("URL:")
    print(f"  {url}")

    print("")
    print("Destination:")
    print(f"  {destination}")

    result = download_gzip(
        url,
        destination,
    )

    print("")
    print(
        f"SUCCESS: {product} @ "
        f"{level_text} km"
    )

    return result


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    print("")
    print("=" * 72)
    print("MRMS VERTICAL DUAL-POL DOWNLOAD")
    print("=" * 72)

    levels = [
        normalize_level(level)
        for level in VERTICAL_DUALPOL_LEVELS_KM
    ]

    print("")
    print(
        "Levels:",
        ", ".join(
            f"{level:.2f} km"
            for level in levels
        ),
    )

    total = 0

    for _, product in VERTICAL_DUALPOL_PRODUCTS.items():
        for level in levels:
            download_level(
                product,
                level,
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
