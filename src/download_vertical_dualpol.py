from __future__ import annotations

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

from download_mrms import download_gzip


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

MAX_ATTEMPTS = 5


# ------------------------------------------------------------
# Product base URLs
# ------------------------------------------------------------

BASE_URLS = {

    "MergedRhoHV":
        MRMS_3D_RHOHV_BASE,

    "MergedZdr":
        MRMS_3D_ZDR_BASE,

}


# ------------------------------------------------------------
# Filename formatting
# ------------------------------------------------------------

def level_string(level_km: float) -> str:
    """
    Convert:

        0.50 -> 00.50
        1.00 -> 01.00
        4.00 -> 04.00
    """

    return f"{level_km:05.2f}"


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

def local_path(
    product: str,
    level_km: float,
) -> Path:

    level = level_string(
        level_km
    )

    directory = (
        DATA_DIR
        / "dualpol"
        / f"{level}km"
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        directory
        / (
            f"MRMS_{product}_"
            f"{level}.latest.grib2"
        )
    )


# ------------------------------------------------------------
# Download
# ------------------------------------------------------------

def download_level(
    product: str,
    level_km: float,
) -> Path:

    level = level_string(
        level_km
    )

    base_url = BASE_URLS[
        product
    ]

    # Example:
    #
    # https://mrms.ncep.noaa.gov/3DRhoHV/
    # MergedRhoHV_00.50/
    # MRMS_MergedRhoHV_00.50.latest.grib2.gz

    directory_url = (
        f"{base_url}/"
        f"{product}_{level}"
    )

    filename = (
        f"MRMS_{product}_"
        f"{level}.latest.grib2.gz"
    )

    url = (
        f"{directory_url}/"
        f"{filename}"
    )

    destination = local_path(
        product,
        level_km,
    )

    print("")
    print(
        f"{product} @ {level:.2f} km"
    )
    print("-" * 72)

    print(
        f"URL:"
    )

    print(
        f"  {url}"
    )

    print(
        f"Destination:"
    )

    print(
        f"  {destination}"
    )

    last_error = None

    for attempt in range(
        1,
        MAX_ATTEMPTS + 1,
    ):

        print(
            f"Attempt "
            f"{attempt}/{MAX_ATTEMPTS}"
        )

        try:

            download_gzip(
                url,
                destination,
            )

            print(
                f"SUCCESS: "
                f"{product} "
                f"{level:.2f} km"
            )

            return destination

        except Exception as exc:

            last_error = exc

            print(
                f"FAILED: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            destination.with_suffix(
                destination.suffix + ".download"
            ).unlink(
                missing_ok=True
            )

            destination.with_suffix(
                destination.suffix + ".tmp"
            ).unlink(
                missing_ok=True
            )

            destination.unlink(
                missing_ok=True
            )

            if attempt < MAX_ATTEMPTS:

                wait = 5 * attempt

                print(
                    f"Retrying in "
                    f"{wait} seconds..."
                )

                time.sleep(
                    wait
                )

    raise RuntimeError(
        f"Unable to download "
        f"{product} at "
        f"{level:.2f} km after "
        f"{MAX_ATTEMPTS} attempts. "
        f"Last error: {last_error}"
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:

    print("")
    print("=" * 72)
    print(
        "MRMS VERTICAL DUAL-POL DOWNLOAD"
    )
    print("=" * 72)

    total = 0

    for _, product in (
        VERTICAL_DUALPOL_PRODUCTS.items()
    ):

        for level_km in (
            VERTICAL_DUALPOL_LEVELS_KM
        ):

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
