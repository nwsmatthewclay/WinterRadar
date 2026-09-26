from __future__ import annotations

"""Build a precipitation-type composite for every historical MRMS scan.

The radar timestamp drives the reflectivity field. RAP pressure-level
profiles are reused by valid hour so the environment can be held constant
between model analyses while the MRMS scan itself changes every ~2 minutes.
A separate process is used by the archive workflow to keep pygrib/ecCodes
bindings isolated from the history decoder.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from classifier import CLEAR, ClassificationResult, rain_intensity_dbz
from download_rap_profile import download_rap_profile, load_rap_profile, sample_profile_to_mrms
from phase_profile import classify_from_vertical_profile, probabilities_to_phase
from read_mrms import get_values
from render import result_to_phase_rgba, result_to_precip_type_rgba

PROFILE_CHUNK_ROWS = 40
DOWNSAMPLE = 4  # keep phase classification lightweight; the final composite uses the full native MRMS reflectivity field


def phase_for_scan(ref, lats, lons, profile):
    # Reduce the expensive profile/radar fusion to a 4x coarser grid, then
    # nearest-neighbor expand the categorical/display result back to the
    # native MRMS grid. The source scan still controls every radar pixel in
    # the composite; only the phase classifier is spatially accelerated.
    ref_c = np.asarray(ref)[::DOWNSAMPLE, ::DOWNSAMPLE]
    if np.asarray(lats).ndim == 1:
        lat_c = np.asarray(lats)[::DOWNSAMPLE]
        lon_c = np.asarray(lons)[::DOWNSAMPLE]
    else:
        lat_c = np.asarray(lats)[::DOWNSAMPLE, ::DOWNSAMPLE]
        lon_c = np.asarray(lons)[::DOWNSAMPLE, ::DOWNSAMPLE]

    phase = np.full(ref_c.shape, CLEAR, dtype=np.uint8)
    confidence = np.zeros(ref_c.shape, dtype=np.float32)
    intensity = np.zeros(ref_c.shape, dtype=np.uint8)

    for y0 in range(0, ref_c.shape[0], PROFILE_CHUNK_ROWS):
        y1 = min(y0 + PROFILE_CHUNK_ROWS, ref_c.shape[0])
        lat_chunk = lat_c[y0:y1] if lat_c.ndim == 1 else lat_c[y0:y1, :]
        lon_chunk = lon_c if lon_c.ndim == 1 else lon_c[y0:y1, :]
        sampled = sample_profile_to_mrms(profile, lat_chunk, lon_chunk)
        probs = classify_from_vertical_profile(
            pressure_hpa=sampled["pressure_hpa"],
            wetbulb_c=sampled["wetbulb_c"],
            height_m=sampled["height_m"],
            temperature_c=sampled["temperature_c"],
            rh_ice_pct=sampled["rh_ice_pct"],
        )
        ph, conf = probabilities_to_phase(probs)
        precip = np.isfinite(ref_c[y0:y1]) & (ref_c[y0:y1] >= 10.0) & sampled["valid"]
        ph[~precip] = CLEAR
        conf[~precip] = 0.0
        phase[y0:y1] = ph
        intensity[y0:y1] = rain_intensity_dbz(ref_c[y0:y1])
        intensity[y0:y1][~precip] = 0

    # Expand only the categorical phase result back to the native MRMS grid.
    # IMPORTANT: render the composite against the ORIGINAL full-resolution
    # reflectivity field, not the 4x-downsampled field. This preserves the
    # exact MRMS texture, intensity gradients, and pixel resolution seen in
    # the raw radar layer. The RAP classifier is the only intentionally
    # downsampled part of the operation.
    native_ref = np.asarray(ref, dtype=np.float32)
    native_h, native_w = native_ref.shape
    phase_native = np.repeat(np.repeat(phase, DOWNSAMPLE, axis=0), DOWNSAMPLE, axis=1)
    phase_native = phase_native[:native_h, :native_w]
    confidence_native = np.repeat(np.repeat(confidence, DOWNSAMPLE, axis=0), DOWNSAMPLE, axis=1)
    confidence_native = confidence_native[:native_h, :native_w]
    intensity_native = np.repeat(np.repeat(intensity, DOWNSAMPLE, axis=0), DOWNSAMPLE, axis=1)
    intensity_native = intensity_native[:native_h, :native_w]

    native_result = ClassificationResult(
        phase=phase_native,
        confidence=confidence_native,
        intensity=intensity_native,
    )
    rgba = result_to_precip_type_rgba(native_result, native_ref)
    phase_rgba = result_to_phase_rgba(native_result)
    return rgba, phase_rgba


def mercator_y(lat_deg):
    max_lat = 85.0511287798
    lat = np.clip(np.asarray(lat_deg, dtype=np.float64), -max_lat, max_lat)
    phi = np.deg2rad(lat)
    return np.log(np.tan(np.pi / 4.0 + phi / 2.0))


def inverse_mercator_lat(y):
    return np.rad2deg(2.0 * np.arctan(np.exp(y)) - np.pi / 2.0)


def project_native_mrms_to_webmercator(rgba, lats, lons):
    """Project a native MRMS lat/lon raster exactly like the live web raster."""
    image = np.asarray(rgba, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 4:
        raise ValueError(f"Expected RGBA image, got {image.shape}")

    lat_arr = np.asarray(lats, dtype=np.float64)
    lon_arr = np.asarray(lons, dtype=np.float64)
    if lat_arr.ndim > 1:
        lat_arr = lat_arr[:, 0]
    if lon_arr.ndim > 1:
        lon_arr = lon_arr[0, :]

    south = float(np.nanmin(lat_arr))
    north = float(np.nanmax(lat_arr))
    if south >= north:
        raise ValueError("Invalid MRMS latitude bounds")

    # MRMS native arrays are north-to-south. If a future reader returns the
    # opposite orientation, normalize it before the projection.
    if lat_arr[0] < lat_arr[-1]:
        image = image[::-1, :, :]

    y_s = float(mercator_y(south))
    y_n = float(mercator_y(north))
    mercator_span = y_n - y_s
    geographic_span_rad = np.deg2rad(north - south)
    output_height = max(2, int(round(image.shape[0] * mercator_span / geographic_span_rad)))

    y_centers = y_n - (np.arange(output_height, dtype=np.float64) + 0.5) * mercator_span / output_height
    lat_centers = inverse_mercator_lat(y_centers)
    source_rows = (north - lat_centers) / (north - south) * (image.shape[0] - 1)
    nearest_rows = np.rint(np.clip(source_rows, 0.0, image.shape[0] - 1)).astype(np.int32)
    return image[nearest_rows, :, :]


def parse_stamp(path: Path) -> datetime:
    stamp = path.stem
    # Expected archive temp names: MRMS_YYYYMMDD-HHMMSS.grib2
    if "_" in stamp:
        stamp = stamp.rsplit("_", 1)[-1]
    return datetime.strptime(stamp, "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    profile_cache: dict[str, dict] = {}
    for raw in args.paths:
        path = Path(raw)
        stamp = parse_stamp(path)
        hour_key = stamp.strftime("%Y%m%d%H")
        print(f"  HIST PHASE: {stamp.isoformat()} -> {path.name}", flush=True)

        profile = profile_cache.get(hour_key)
        if profile is None:
            profile_path = download_rap_profile(stamp.isoformat())
            profile = load_rap_profile(profile_path)
            profile_cache[hour_key] = profile
            print(
                f"  HIST PHASE RAP: using {profile.get('valid_time_utc')} for {hour_key}Z scans",
                flush=True,
            )

        ref, lats, lons = get_values(path, product="MergedReflectivityQCComposite")
        rgba, phase_rgba = phase_for_scan(ref, lats, lons, profile)

        # The browser map is Web Mercator. Do the same native->Web Mercator
        # row projection used by the live MRMS raster so the historical
        # composite overlays the radar exactly instead of being stretched as
        # a geographic lat/lon image inside a Web Mercator map.
        rgba_web = project_native_mrms_to_webmercator(rgba, lats, lons)
        phase_web = project_native_mrms_to_webmercator(phase_rgba, lats, lons)

        phase_name = f"phase_conus_{stamp:%Y%m%d-%H%M%S}.webp"
        Image.fromarray(phase_web, mode="RGBA").save(
            output_dir / phase_name,
            format="WEBP",
            quality=95,
            method=6,
        )
        out_name = f"preciptype_conus_{stamp:%Y%m%d-%H%M%S}.webp"
        Image.fromarray(rgba_web, mode="RGBA").save(
            output_dir / out_name,
            format="WEBP",
            quality=95,
            method=6,
        )
        print(f"  HIST PHASE READY: {out_name} ({(output_dir / out_name).stat().st_size:,} bytes)", flush=True)


if __name__ == "__main__":
    main()
