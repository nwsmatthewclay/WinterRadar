from __future__ import annotations

import gzip
import shutil
from pathlib import Path

import requests

from config import DATA_DIR, MRMS_BASE, PRODUCTS


def download_latest(product: str) -> Path:
    """Download the latest operational MRMS GRIB2 field."""
    url = f"{MRMS_BASE}/{product}/MRMS_{product}.latest.grib2.gz"
    gz_path = DATA_DIR / f"MRMS_{product}.latest.grib2.gz"
    grib_path = DATA_DIR / f"MRMS_{product}.latest.grib2"

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with requests.get(url, timeout=90, stream=True) as response:
        response.raise_for_status()
        with gz_path.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)

    with gzip.open(gz_path, "rb") as src, grib_path.open("wb") as dst:
        shutil.copyfileobj(src, dst)

    return grib_path


def main() -> None:
    for name, product in PRODUCTS.items():
        print(f"Downloading {name}: {product}")
        path = download_latest(product)
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
