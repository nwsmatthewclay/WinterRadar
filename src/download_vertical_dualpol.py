from __future__ import annotations

import time
from pathlib import Path

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
# MRMS 3-D product locations
# ------------------------------------------------------------

BASE_URLS = {
    "MergedRhoHV": MRMS_3D_RHOHV_BASE,
    "MergedZdr": MRMS_3D_ZDR_BASE,
}


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def normalize_level(level) -> float:
    """
    Force configuration values such as:

        0.50
        "0.50"
        "00.50"

    into a proper float.
    """

    try:
        return float(level)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid MRMS vertical level: {level!r}"
        ) from exc


def level_string(level) -> str:
    """
    Format an MRMS CAPPI level.

    Examples:

        0.50 -> 00.50
        1.00 -> 01.00
        4.00 -> 04.00
    """

    level = normalize_level(level)

    return f"{level:05.2f}"


def local_path(
    product: str,
    level,
) -> Path:

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
        / (
            f"MRMS_{product}_"
            f"{level_text}.latest.grib2"
        )
    )


# ------------------------------------------------------------
# Download one CAPPI
# ------------------------------------------------------------

def download_level(
    product: str,
    level,
) -> Path:

    level = normalize_level(level)
    level_text = level_string(level)

    print("")
    print("=" * 72)
    print(
        f"{product} @ {level_text} km"
    )
    print("=" * 72)

    if product not in BASE_URLS:
        raise ValueError(
            f"No 3-D MRMS base URL configured "
            f"for {product}"
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

    print(f"URL:")
    print(f"  {url}")

    print("")
    print(f"Destination:")
    print(f"  {destination}")

    last_error = None

    for attempt in range(
        1,
        MAX_ATTEMPTS + 1,
    ):

        print("")
        print(
            f"Attempt "
            f"{attempt}/{MAX_ATTEMPTS}"
        )

        try:

            download_gzip(
                url,
                destination,
            )

            print("")
            print(
                f"SUCCESS: "
                f"{product} @ "
                f"{level_text} km"
            )

            return destination

        except Exception as exc:

            last_error = exc

            print("")
            print(
                f"FAILED: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            # Remove failed partial files.
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

                wait_seconds = (
                    5 * attempt
                )

                print(
                    f"Retrying in "
                    f"{wait_seconds} seconds..."
                )

                time.sleep(
                    wait_seconds
                )

    raise RuntimeError(
        f"Unable to download "
        f"{product} @ "
        f"{level_text} km after "
        f"{MAX_ATTEMPTS} attempts. "
        f"Last error: {last_error}"
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("")
    print("=" * 72)
    print(
        "MRMS VERTICAL DUAL-POL DOWNLOAD"
    )
    print("=" * 72)

    # Normalize levels once at startup.
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

    for _, product in (
        VERTICAL_DUALPOL_PRODUCTS.items()
    ):

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
