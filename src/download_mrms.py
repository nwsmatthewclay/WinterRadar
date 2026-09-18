from __future__ import annotations

import gzip
import shutil
import time
from pathlib import Path

import requests

from config import DATA_DIR, MRMS_BASE, PRODUCTS


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

MAX_ATTEMPTS = 5

CONNECT_TIMEOUT = 20
READ_TIMEOUT = 120

CHUNK_SIZE = 1024 * 1024  # 1 MB

USER_AGENT = (
    "WinterRadar/1.0 "
    "(MRMS winter precipitation visualization project)"
)


# ------------------------------------------------------------
# HTTP session
# ------------------------------------------------------------

SESSION = requests.Session()

SESSION.headers.update(
    {
        "User-Agent": USER_AGENT,
        "Accept": "*/*",
        "Accept-Encoding": "identity",
        "Connection": "keep-alive",
    }
)


# ------------------------------------------------------------
# Download helpers
# ------------------------------------------------------------

def download_gzip(url: str, destination: Path) -> None:
    """
    Download a gzip file to a temporary path.

    The file is not considered successful until the complete
    gzip stream has been written.
    """

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_gz = destination.with_suffix(
        destination.suffix + ".download"
    )

    # Remove any debris from an earlier failed attempt.
    temp_gz.unlink(
        missing_ok=True
    )

    print(f"Downloading:")
    print(f"  {url}")
    print(f"  -> {temp_gz}")

    with SESSION.get(
        url,
        stream=True,
        timeout=(
            CONNECT_TIMEOUT,
            READ_TIMEOUT,
        ),
    ) as response:

        response.raise_for_status()

        expected_size = response.headers.get(
            "Content-Length"
        )

        if expected_size:
            expected_size = int(expected_size)

            print(
                f"  Expected compressed size: "
                f"{expected_size:,} bytes"
            )

        bytes_written = 0

        with temp_gz.open(
            "wb"
        ) as output:

            for chunk in response.iter_content(
                chunk_size=CHUNK_SIZE
            ):

                if not chunk:
                    continue

                output.write(chunk)
                bytes_written += len(chunk)

    print(
        f"  Downloaded: "
        f"{bytes_written:,} bytes"
    )

    # --------------------------------------------------------
    # Basic Content-Length check
    # --------------------------------------------------------

    if (
        expected_size is not None
        and bytes_written != expected_size
    ):
        raise IOError(
            "Downloaded file size does not match "
            f"Content-Length "
            f"({bytes_written} != {expected_size})"
        )

    # --------------------------------------------------------
    # Validate the gzip stream completely
    # --------------------------------------------------------

    print("  Checking gzip integrity...")

    decompressed_bytes = 0

    with gzip.open(
        temp_gz,
        "rb",
    ) as source:

        while True:
            chunk = source.read(
                CHUNK_SIZE
            )

            if not chunk:
                break

            decompressed_bytes += len(chunk)

    if decompressed_bytes <= 0:
        raise IOError(
            "Gzip stream was valid but contained "
            "no decompressed data."
        )

    print(
        f"  Gzip OK: "
        f"{decompressed_bytes:,} decompressed bytes"
    )

    # --------------------------------------------------------
    # Decompress to another temporary file
    # --------------------------------------------------------

    temp_grib = destination.with_suffix(
        destination.suffix + ".tmp"
    )

    temp_grib.unlink(
        missing_ok=True
    )

    print(
        f"  Decompressing -> {temp_grib}"
    )

    with gzip.open(
        temp_gz,
        "rb",
    ) as source, temp_grib.open(
        "wb"
    ) as target:

        shutil.copyfileobj(
            source,
            target,
            length=CHUNK_SIZE,
        )

    # --------------------------------------------------------
    # Atomically replace the final GRIB2 file
    # --------------------------------------------------------

    temp_grib.replace(
        destination
    )

    temp_gz.unlink(
        missing_ok=True
    )

    print(
        f"  Final file: {destination}"
    )


def download_latest(
    product: str,
) -> Path:

    url = (
        f"{MRMS_BASE}/{product}/"
        f"MRMS_{product}.latest.grib2.gz"
    )

    destination = (
        DATA_DIR
        / f"MRMS_{product}.latest.grib2"
    )

    last_error = None

    for attempt in range(
        1,
        MAX_ATTEMPTS + 1,
    ):

        print("")
        print(
            f"Attempt {attempt}/{MAX_ATTEMPTS} "
            f"for {product}"
        )

        # Never allow an old partial output to survive.
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

        try:

            download_gzip(
                url,
                destination,
            )

            print(
                f"SUCCESS: {product}"
            )

            return destination

        except Exception as exc:

            last_error = exc

            print(
                f"FAILED: {product}"
            )

            print(
                f"Reason: {type(exc).__name__}: {exc}"
            )

            # Clean up anything from the failed attempt.
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

                wait_seconds = 5 * attempt

                print(
                    f"Retrying in "
                    f"{wait_seconds} seconds..."
                )

                time.sleep(
                    wait_seconds
                )

    raise RuntimeError(
        f"Unable to download valid MRMS product "
        f"{product} after {MAX_ATTEMPTS} attempts. "
        f"Last error: {last_error}"
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("")
    print("=" * 72)
    print("MRMS DOWNLOAD")
    print("=" * 72)

    successful = 0

    for name, product in PRODUCTS.items():

        print("")
        print(
            f"Downloading {name}: {product}"
        )

        try:

            path = download_latest(
                product
            )

            print(
                f"  -> {path}"
            )

            successful += 1

        except Exception as exc:

            print("")
            print(
                f"ERROR downloading "
                f"{product}: {exc}"
            )

            # Fail the complete run rather than allowing
            # one missing field to contaminate the classifier.
            raise

    print("")
    print("=" * 72)
    print(
        f"MRMS DOWNLOAD COMPLETE: "
        f"{successful}/{len(PRODUCTS)} fields"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
