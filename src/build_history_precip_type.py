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
DOWNSAMPLE = 4


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

    result = ClassificationResult(phase=phase, confidence=confidence, intensity=intensity)
    rgba_coarse = result_to_precip_type_rgba(result, ref_c)
    phase_coarse = result_to_phase_rgba(result)

    # Restore native scan dimensions. This keeps the exact scan's dBZ field in
    # the hue/intensity calculation at the classifier grid while avoiding a
    # full 24.5-million-cell RAP nearest-neighbor operation for every scan.
    native_h, native_w = np.asarray(ref).shape
    rgba = np.repeat(np.repeat(rgba_coarse, DOWNSAMPLE, axis=0), DOWNSAMPLE, axis=1)
    phase_rgba = np.repeat(np.repeat(phase_coarse, DOWNSAMPLE, axis=0), DOWNSAMPLE, axis=1)
    return rgba[:native_h, :native_w], phase_rgba[:native_h, :native_w]


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
        phase_name = f"phase_conus_{stamp:%Y%m%d-%H%M%S}.webp"
        Image.fromarray(phase_rgba, mode="RGBA").save(
            output_dir / phase_name,
            format="WEBP",
            quality=88,
            method=4,
        )
        out_name = f"preciptype_conus_{stamp:%Y%m%d-%H%M%S}.webp"
        Image.fromarray(rgba, mode="RGBA").save(
            output_dir / out_name,
            format="WEBP",
            quality=88,
            method=4,
        )
        print(f"  HIST PHASE READY: {out_name} ({(output_dir / out_name).stat().st_size:,} bytes)", flush=True)


if __name__ == "__main__":
    main()
