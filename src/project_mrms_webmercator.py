from __future__ import annotations

"""Project the native MRMS PNGs from EPSG:4326 row spacing to Web Mercator.

The MRMS source image is on a regular geographic lat/lon grid. Leaflet's
standard CRS is EPSG:3857, so an unwarped geographic raster will appear
vertically displaced when it is used directly with L.imageOverlay.

This utility deliberately runs AFTER the live MRMS core has completed. It
never changes the native MRMS outputs. It creates separate *_web.png files
for the browser.
"""

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"

WEB_VERSION = "1.0-webmercator-postprocess"
MAX_LAT = 85.0511287798


def mercator_y(lat_deg: np.ndarray | float) -> np.ndarray | float:
    lat = np.clip(np.asarray(lat_deg, dtype=np.float64), -MAX_LAT, MAX_LAT)
    phi = np.deg2rad(lat)
    return np.log(np.tan(np.pi / 4.0 + phi / 2.0))


def inverse_mercator_lat(y: np.ndarray) -> np.ndarray:
    return np.rad2deg(2.0 * np.arctan(np.exp(y)) - np.pi / 2.0)


def validate_bounds(bounds: list[float]) -> tuple[float, float, float, float]:
    if len(bounds) != 4:
        raise ValueError(f"Expected four bounds values, got {bounds!r}")
    south, west, north, east = map(float, bounds)
    if not (-90 < south < north < 90):
        raise ValueError(f"Invalid latitude bounds: {bounds!r}")
    if not (-180 <= west < east <= 180):
        raise ValueError(f"Invalid longitude bounds: {bounds!r}")
    return south, west, north, east


def source_row_for_lat(lat: np.ndarray, south: float, north: float, height: int) -> np.ndarray:
    # MRMS latitude rows are descending (north at row 0). Use continuous row
    # coordinates so the warp does not accumulate a rounding bias.
    row = (north - lat) / (north - south) * (height - 1)
    return np.clip(row, 0.0, height - 1.0)


def project_image(src_path: Path, dst_path: Path, bounds: tuple[float, float, float, float]) -> dict:
    south, west, north, east = bounds
    with Image.open(src_path) as im:
        src = np.asarray(im.convert("RGBA"))

    src_h, src_w = src.shape[:2]
    if src_h < 2 or src_w < 2:
        raise ValueError(f"Image too small to project: {src.shape}")

    # Longitude is already linear in EPSG:4326 and EPSG:3857, so retain the
    # native width. Only the vertical axis needs nonlinear resampling.
    y_s = float(mercator_y(south))
    y_n = float(mercator_y(north))
    mercator_span = y_n - y_s
    geographic_span_rad = math.radians(north - south)
    output_h = max(2, int(round(src_h * mercator_span / geographic_span_rad)))

    # Output row centers are uniformly spaced in Mercator Y. Work in chunks
    # so the 3500x7000 MRMS raster never requires several extra full-size
    # temporary arrays at once.
    out = np.empty((output_h, src_w, 4), dtype=np.uint8)
    y_centers = y_n - (np.arange(output_h, dtype=np.float64) + 0.5) * mercator_span / output_h
    lat_centers = inverse_mercator_lat(y_centers)
    source_rows = source_row_for_lat(lat_centers, south, north, src_h)
    nearest_rows = np.rint(source_rows).astype(np.int32)
    nearest_rows = np.clip(nearest_rows, 0, src_h - 1)
    out[:] = src[nearest_rows]

    Image.fromarray(out, mode="RGBA").save(dst_path, optimize=True)

    return {
        "source_width": src_w,
        "source_height": src_h,
        "output_width": src_w,
        "output_height": output_h,
        "south": south,
        "west": west,
        "north": north,
        "east": east,
        "projection": "EPSG:3857-webmercator-image",
        "webmercator_y_south": y_s,
        "webmercator_y_north": y_n,
    }


def main() -> None:
    metadata_path = OUTPUT_DIR / "mrms_current.json"
    if not metadata_path.exists():
        raise SystemExit("Missing outputs/mrms_current.json")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    bounds = validate_bounds(metadata["bounds"])

    print("=" * 72)
    print(f"MRMS WEB MERCATOR POST-PROCESSOR — {WEB_VERSION}")
    print("Native MRMS outputs are preserved; only browser copies are projected.")
    print("=" * 72)
    print(f"Geographic bounds: {bounds}")

    radar_info = project_image(
        OUTPUT_DIR / "mrms_current.png",
        OUTPUT_DIR / "mrms_current_web.png",
        bounds,
    )
    phase_info = project_image(
        OUTPUT_DIR / "winter_phase_mask.png",
        OUTPUT_DIR / "winter_phase_mask_web.png",
        bounds,
    )

    web_metadata = dict(metadata)
    web_metadata.update(
        {
            "web_projection": "EPSG:3857",
            "web_image_origin": "upper",
            "web_image_bounds": list(bounds),
            "web_version": WEB_VERSION,
            "web_radar_size": [radar_info["output_width"], radar_info["output_height"]],
            "web_phase_size": [phase_info["output_width"], phase_info["output_height"]],
            "webmercator_y_south": radar_info["webmercator_y_south"],
            "webmercator_y_north": radar_info["webmercator_y_north"],
        }
    )
    (OUTPUT_DIR / "mrms_web.json").write_text(
        json.dumps(web_metadata, indent=2), encoding="utf-8"
    )

    for name in ("mrms_current_web.png", "winter_phase_mask_web.png", "mrms_web.json"):
        path = OUTPUT_DIR / name
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Projected output missing or empty: {path}")
        print(f"  {path.name}: {path.stat().st_size:,} bytes")

    print("WEB MERCATOR PROJECTION QC PASSED")


if __name__ == "__main__":
    main()
