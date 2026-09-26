from __future__ import annotations

import argparse
import gzip
import os
import shutil
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import (
    DATA_DIR,
    MRMS_BASE,
    LIVE_PRODUCTS,
    OPTIONAL_DIAGNOSTIC_PRODUCTS,
    REQUIRED_LIVE_PRODUCTS,
    REQUIRED_ATTEMPTS,
    REQUIRED_CONNECT_TIMEOUT,
    REQUIRED_READ_TIMEOUT,
    OPTIONAL_ATTEMPTS,
    OPTIONAL_CONNECT_TIMEOUT,
    OPTIONAL_READ_TIMEOUT,
)

USER_AGENT = "WinterRadar/1.0 (NWS MRMS operational visualization)"
RETRYABLE_STATUS = (408, 429, 500, 502, 503, 504)

def _session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=0,
        connect=0,
        read=0,
        redirect=2,
        status=0,
        backoff_factor=0,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "*/*"})
    return session


def _validate_grib(path: Path) -> None:
    if not path.exists() or path.stat().st_size < 16:
        raise RuntimeError(f"Downloaded GRIB is missing or too small: {path}")

    with path.open("rb") as fh:
        magic = fh.read(4)
    if magic != b"GRIB":
        raise RuntimeError(
            f"Downloaded file is not a GRIB2 payload (starts with {magic!r}): {path}"
        )


def _download_to_gz(
    session: requests.Session,
    url: str,
    gz_path: Path,
    *,
    attempts: int,
    connect_timeout: int,
    read_timeout: int,
) -> None:
    part = gz_path.with_name(gz_path.name + ".part")
    if part.exists():
        part.unlink()

    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            print(f"  HTTP GET attempt {attempt}/{attempts}")
            with session.get(
                url,
                timeout=(connect_timeout, read_timeout),
                stream=True,
            ) as response:
                if response.status_code in RETRYABLE_STATUS:
                    raise requests.HTTPError(
                        f"Transient HTTP {response.status_code}",
                        response=response,
                    )
                response.raise_for_status()

                with part.open("wb") as fh:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            fh.write(chunk)

            if part.stat().st_size < 32:
                raise RuntimeError(f"Downloaded payload is unexpectedly small: {part}")

            # Validate the compressed stream before making it the live file.
            with gzip.open(part, "rb") as fh:
                magic = fh.read(4)
            if magic != b"GRIB":
                raise RuntimeError(
                    f"Compressed response does not contain a GRIB payload: {url}"
                )

            os.replace(part, gz_path)
            return

        except Exception as exc:
            last_error = exc
            if part.exists():
                part.unlink()
            if attempt < attempts:
                delay = 2 ** (attempt - 1)
                print(f"  Download failed: {exc}; retrying in {delay}s")
                time.sleep(delay)
            else:
                break

    raise RuntimeError(f"Failed to download {url}: {last_error}")


def _decompress_atomic(gz_path: Path, grib_path: Path) -> None:
    part = grib_path.with_name(grib_path.name + ".part")
    if part.exists():
        part.unlink()

    try:
        with gzip.open(gz_path, "rb") as src, part.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
        _validate_grib(part)
        os.replace(part, grib_path)
    finally:
        if part.exists():
            part.unlink()


def download_product(product: str, *, required: bool) -> Path | None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    url = f"{MRMS_BASE}/{product}/MRMS_{product}.latest.grib2.gz"
    gz_path = DATA_DIR / f"MRMS_{product}.latest.grib2.gz"
    grib_path = DATA_DIR / f"MRMS_{product}.latest.grib2"

    print(f"Downloading {product} ({'required' if required else 'optional'})")

    try:
        with _session() as session:
            _download_to_gz(
                session,
                url,
                gz_path,
                attempts=REQUIRED_ATTEMPTS if required else OPTIONAL_ATTEMPTS,
                connect_timeout=REQUIRED_CONNECT_TIMEOUT if required else OPTIONAL_CONNECT_TIMEOUT,
                read_timeout=REQUIRED_READ_TIMEOUT if required else OPTIONAL_READ_TIMEOUT,
            )
        _decompress_atomic(gz_path, grib_path)
        print(f"  OK -> {grib_path} ({grib_path.stat().st_size:,} bytes)")

        # Compatibility alias for older diagnostic/fusion scripts that still
        # request the former reflectivity filename.  The bytes are the
        # authoritative MergedReflectivityComposite field; no second radar
        # download is performed.
        if product == "MergedReflectivityComposite":
            legacy = DATA_DIR / "MRMS_ReflectivityAtLowestAltitude.latest.grib2"
            shutil.copyfile(grib_path, legacy)
            print(f"  Compatibility alias -> {legacy}")

        return grib_path
    except Exception as exc:
        if required:
            raise
        print(f"  OPTIONAL DOWNLOAD FAILED: {exc}")
        return None


def download_required_live_products() -> dict[str, Path | None]:
    """Download only the hard-required live radar product."""
    return {
        "reflectivity": download_product(
            REQUIRED_LIVE_PRODUCTS["reflectivity"],
            required=True,
        )
    }


def download_optional_live_products() -> dict[str, Path | None]:
    """Best-effort phase-support fields for the live map."""
    results: dict[str, Path | None] = {}
    for name, product in LIVE_PRODUCTS.items():
        if name in REQUIRED_LIVE_PRODUCTS:
            continue
        results[name] = download_product(product, required=False)
    return results


def download_live_products() -> dict[str, Path | None]:
    """Backward-compatible wrapper: required first, then optional."""
    results = download_required_live_products()
    results.update(download_optional_live_products())
    return results

def download_diagnostic_products() -> dict[str, Path | None]:
    """Best-effort download of non-critical 2-D diagnostic fields."""
    results: dict[str, Path | None] = {}
    for name, product in OPTIONAL_DIAGNOSTIC_PRODUCTS.items():
        results[name] = download_product(product, required=False)
    return results


def main(include_diagnostics: bool = False) -> None:
    print("=" * 72)
    print("MRMS DOWNLOAD")
    print("=" * 72)

    live = download_live_products()

    if include_diagnostics:
        print("=" * 72)
        print("OPTIONAL DIAGNOSTIC DOWNLOADS")
        print("=" * 72)
        download_diagnostic_products()

    missing = [
        name for name, path in live.items() if path is None and name not in REQUIRED_LIVE_PRODUCTS
    ]
    if missing:
        print("Live phase-support fields unavailable: " + ", ".join(missing))

    print("MRMS DOWNLOAD COMPLETE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Also download best-effort non-critical diagnostic 2-D products.",
    )
    args = parser.parse_args()
    main(include_diagnostics=args.diagnostics)
