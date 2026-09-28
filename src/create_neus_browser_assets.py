from __future__ import annotations

"""Create small NEUS browser copies from the full-CONUS MRMS rasters."""

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from config import NEUS_BOUNDS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
MAX_LAT = 85.0511287798


def mercator_y(lat_deg):
    lat = np.clip(np.asarray(lat_deg, dtype=np.float64), -MAX_LAT, MAX_LAT)
    return np.log(np.tan(np.pi / 4.0 + np.deg2rad(lat) / 2.0))


def inverse_mercator_lat(y):
    return np.rad2deg(2.0 * np.arctan(np.exp(y)) - np.pi / 2.0)


def project_array(src, bounds):
    south, west, north, east = bounds
    h, w = src.shape[:2]
    y_s = float(mercator_y(south))
    y_n = float(mercator_y(north))
    span = y_n - y_s
    output_h = max(2, int(round(h * span / math.radians(north - south))))
    y_centers = y_n - (np.arange(output_h, dtype=np.float64) + 0.5) * span / output_h
    lat_centers = inverse_mercator_lat(y_centers)
    rows = (north - lat_centers) / (north - south) * (h - 1)
    rows = np.rint(np.clip(rows, 0, h - 1)).astype(np.int32)
    return src[rows]


def crop_source(src, full_bounds, target_bounds):
    fsouth, fwest, fnorth, feast = full_bounds
    south, west, north, east = target_bounds
    if not (fsouth <= south < north <= fnorth and fwest <= west < east <= feast):
        raise ValueError(f"NEUS bounds {target_bounds} are outside MRMS bounds {full_bounds}")

    h, w = src.shape[:2]
    y0 = int(math.floor((fnorth - north) / (fnorth - fsouth) * (h - 1)))
    y1 = int(math.ceil((fnorth - south) / (fnorth - fsouth) * (h - 1))) + 1
    x0 = int(math.floor((west - fwest) / (feast - fwest) * (w - 1)))
    x1 = int(math.ceil((east - fwest) / (feast - fwest) * (w - 1))) + 1
    y0 = max(0, min(h - 1, y0))
    y1 = max(y0 + 1, min(h, y1))
    x0 = max(0, min(w - 1, x0))
    x1 = max(x0 + 1, min(w, x1))
    return src[y0:y1, x0:x1, :]


def make_one(source_name, output_name, full_bounds, target_bounds):
    source = OUT / source_name
    with Image.open(source) as im:
        src = np.asarray(im.convert("RGBA"))
    cropped = crop_source(src, full_bounds, target_bounds)
    projected = project_array(cropped, target_bounds)
    png = OUT / output_name
    Image.fromarray(projected, "RGBA").save(png, optimize=True)
    Image.fromarray(projected, "RGBA").save(
        png.with_suffix(".webp"), format="WEBP", lossless=True, method=6
    )
    print(f"  {png.name}: {projected.shape[1]}x{projected.shape[0]}")
    print(f"    PNG  {png.stat().st_size:,} bytes")
    print(f"    WebP {png.with_suffix('.webp').stat().st_size:,} bytes")


def main():
    meta_path = OUT / "mrms_current.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    full_bounds = tuple(float(v) for v in metadata["bounds"])
    target_bounds = tuple(float(v) for v in NEUS_BOUNDS)

    specs = (
        ("mrms_current.png", "mrms_current_regional_web.png"),
        ("winter_phase_mask.png", "winter_phase_mask_regional_web.png"),
        ("winter_precip_type.png", "winter_precip_type_regional_web.png"),
        ("winter_radar_composite.png", "winter_radar_composite_regional_web.png"),
    )

    for source_name, output_name in specs:
        make_one(source_name, output_name, full_bounds, target_bounds)

    regional = {
        "bounds": list(target_bounds),
        "bounds_format": ["south", "west", "north", "east"],
        "projection": "EPSG:3857",
        "source_domain": "Full-CONUS MRMS",
        "presentation_domain": "Northeast U.S.",
    }
    (OUT / "mrms_regional_web.json").write_text(
        json.dumps(regional, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
