from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from classifier import classify_initial
from config import (
    DATA_DIR,
    OUTPUT_DIR,
    PHASE_COLORS,
    SPATIAL_REFERENCE,
)
from download_mrms import (
    download_all,
    get_mrms_valid_time,
)
from download_rap_profile import (
    download_rap_profile,
    sample_profile_to_mrms,
)
from phase_profile import classify_from_vertical_profile
from read_mrms import (
    get_values,
    get_values_with_time,
    read_grib,
)
from render import (
    result_to_phase_rgba,
    result_to_precip_type_rgba,
    save_rgba_png,
)


MAIN_VERSION = "9.4-rap-profile-phase-sampler"

PROFILE_CHUNK_ROWS = 40

DIAG_Y_FACTOR = 10
DIAG_X_FACTOR = 10


def _path_for(name: str) -> Path:
    mapping = {
        "reflectivity": DATA_DIR
        / "MRMS_ReflectivityAtLowestAltitude.latest.grib2",
        "precip_flag": DATA_DIR
        / "MRMS_PrecipFlag.latest.grib2",
        "bb_top": DATA_DIR
        / "MRMS_BrightBandTopHeight.latest.grib2",
        "bb_bottom": DATA_DIR
        / "MRMS_BrightBandBottomHeight.latest.grib2",
        "rqi": DATA_DIR
        / "MRMS_RadarQualityIndex.latest.grib2",
        "wetbulb": DATA_DIR
        / "MRMS_Model_WetBulbTemp.latest.grib2",
    }

    if name not in mapping:
        raise KeyError(name)

    return mapping[name]


def _load_field(
    name: str,
    shape: tuple[int, int],
) -> np.ndarray:
    path = _path_for(name)

    if not path.exists():
        return np.full(shape, np.nan, dtype=np.float32)

    try:
        values = get_values(path)
    except Exception:
        return np.full(shape, np.nan, dtype=np.float32)

    values = np.asarray(values, dtype=np.float32)

    if values.shape != shape:
        return np.full(shape, np.nan, dtype=np.float32)

    return values


def try_load_optional(
    name: str,
    shape: tuple[int, int],
    default: float = np.nan,
) -> np.ndarray:
    path = _path_for(name)

    if not path.exists():
        return np.full(shape, default, dtype=np.float32)

    try:
        values = get_values(path)
    except Exception as exc:
        print(f"  WARNING: unable to load optional field {name}: {exc}")
        return np.full(shape, default, dtype=np.float32)

    values = np.asarray(values, dtype=np.float32)

    if values.shape != shape:
        print(
            f"  WARNING: optional field {name} shape mismatch: "
            f"{values.shape} != {shape}"
        )
        return np.full(shape, default, dtype=np.float32)

    return values


def _normalize_longitudes(lons: np.ndarray) -> np.ndarray:
    lons = np.asarray(lons, dtype=np.float64)
    return ((lons + 180.0) % 360.0) - 180.0


def _build_diagnostic_probability_grid(
    probability_data: dict[str, np.ndarray],
    source_shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    diag_h = max(1, source_shape[0] // DIAG_Y_FACTOR)
    diag_w = max(1, source_shape[1] // DIAG_X_FACTOR)

    output: dict[str, np.ndarray] = {}

    for key, values in probability_data.items():
        values = np.asarray(values, dtype=np.float32)

        if values.ndim != 2:
            output[key] = np.full(
                (diag_h, diag_w),
                np.nan,
                dtype=np.float32,
            )
            continue

        y_edges = np.linspace(
            0,
            values.shape[0],
            diag_h + 1,
            dtype=int,
        )
        x_edges = np.linspace(
            0,
            values.shape[1],
            diag_w + 1,
            dtype=int,
        )

        diag = np.full(
            (diag_h, diag_w),
            np.nan,
            dtype=np.float32,
        )

        for dy in range(diag_h):
            y0 = y_edges[dy]
            y1 = y_edges[dy + 1]

            if y1 <= y0:
                y1 = min(values.shape[0], y0 + 1)

            for dx in range(diag_w):
                x0 = x_edges[dx]
                x1 = x_edges[dx + 1]

                if x1 <= x0:
                    x1 = min(values.shape[1], x0 + 1)

                block = values[y0:y1, x0:x1]

                if block.size == 0:
                    continue

                finite = block[np.isfinite(block)]

                if finite.size:
                    diag[dy, dx] = np.nanmean(finite)

        output[key] = diag

    return output


def _profile_phase_result(
    ref: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    mrms_time_utc: datetime,
):
    profile = download_rap_profile(mrms_time_utc)

    sample = sample_profile_to_mrms(
        profile,
        lats,
        lons,
    )

    wetbulb = np.asarray(
        sample["wetbulb_c"],
        dtype=np.float32,
    )

    temperature = np.asarray(
        sample["temperature_c"],
        dtype=np.float32,
    )

    rh_ice = np.asarray(
        sample["rh_ice_pct"],
        dtype=np.float32,
    )

    height = np.asarray(
        sample["height_m"],
        dtype=np.float32,
    )

    pressure = np.asarray(
        sample["pressure_hpa"],
        dtype=np.float32,
    )

    valid = np.asarray(
        sample["valid"],
        dtype=bool,
    )

    precip_mask = np.isfinite(ref) & (ref >= 5.0)

    sample_valid_cells = int(np.count_nonzero(valid))
    precip_overlap_cells = int(
        np.count_nonzero(valid & precip_mask)
    )

    phase = np.full(
        ref.shape,
        "CLEAR",
        dtype=object,
    )

    rain_probability = np.full(
        ref.shape,
        np.nan,
        dtype=np.float32,
    )

    snow_probability = np.full(
        ref.shape,
        np.nan,
        dtype=np.float32,
    )

    sleet_probability = np.full(
        ref.shape,
        np.nan,
        dtype=np.float32,
    )

    freezing_rain_probability = np.full(
        ref.shape,
        np.nan,
        dtype=np.float32,
    )

    melting_energy = np.full(
        ref.shape,
        np.nan,
        dtype=np.float32,
    )

    refreezing_energy = np.full(
        ref.shape,
        np.nan,
        dtype=np.float32,
    )

    prob_ice = np.full(
        ref.shape,
        np.nan,
        dtype=np.float32,
    )

    full_profile_cells = 0
    phase_cells = 0

    diag_h = max(
        1,
        ref.shape[0] // DIAG_Y_FACTOR,
    )

    diag_w = max(
        1,
        ref.shape[1] // DIAG_X_FACTOR,
    )

    diagnostic_valid = np.zeros(
        (diag_h, diag_w),
        dtype=bool,
    )

    diagnostic_probability_data = {
        "rain": np.full(
            (diag_h, diag_w),
            np.nan,
            dtype=np.float32,
        ),
        "snow": np.full(
            (diag_h, diag_w),
            np.nan,
            dtype=np.float32,
        ),
        "sleet": np.full(
            (diag_h, diag_w),
            np.nan,
            dtype=np.float32,
        ),
        "freezing_rain": np.full(
            (diag_h, diag_w),
            np.nan,
            dtype=np.float32,
        ),
        "melting_energy": np.full(
            (diag_h, diag_w),
            np.nan,
            dtype=np.float32,
        ),
        "refreezing_energy": np.full(
            (diag_h, diag_w),
            np.nan,
            dtype=np.float32,
        ),
        "prob_ice": np.full(
            (diag_h, diag_w),
            np.nan,
            dtype=np.float32,
        ),
    }

    for y0 in range(
        0,
        ref.shape[0],
        PROFILE_CHUNK_ROWS,
    ):
        y1 = min(
            ref.shape[0],
            y0 + PROFILE_CHUNK_ROWS,
        )

        for y in range(y0, y1):
            for x in range(ref.shape[1]):
                if not precip_mask[y, x]:
                    continue

                if not valid[y, x]:
                    continue

                profile_valid = (
                    np.isfinite(wetbulb[:, y, x])
                    & np.isfinite(temperature[:, y, x])
                    & np.isfinite(rh_ice[:, y, x])
                    & np.isfinite(height[:, y, x])
                    & np.isfinite(pressure)
                )

                if np.count_nonzero(profile_valid) < 4:
                    continue

                full_profile_cells += 1

                try:
                    result = classify_from_vertical_profile(
                        pressure_hpa=pressure[profile_valid],
                        temperature_c=temperature[
                            profile_valid,
                            y,
                            x,
                        ],
                        wetbulb_c=wetbulb[
                            profile_valid,
                            y,
                            x,
                        ],
                        height_m=height[
                            profile_valid,
                            y,
                            x,
                        ],
                        rh_ice_pct=rh_ice[
                            profile_valid,
                            y,
                            x,
                        ],
                    )
                except Exception:
                    continue

                phase_name = result.get(
                    "phase",
                    "CLEAR",
                )

                phase[y, x] = phase_name
                phase_cells += 1

                rain_probability[y, x] = result.get(
                    "rain_probability",
                    np.nan,
                )

                snow_probability[y, x] = result.get(
                    "snow_probability",
                    np.nan,
                )

                sleet_probability[y, x] = result.get(
                    "sleet_probability",
                    np.nan,
                )

                freezing_rain_probability[y, x] = result.get(
                    "freezing_rain_probability",
                    np.nan,
                )

                melting_energy[y, x] = result.get(
                    "melting_energy",
                    np.nan,
                )

                refreezing_energy[y, x] = result.get(
                    "refreezing_energy",
                    np.nan,
                )

                prob_ice[y, x] = result.get(
                    "prob_ice",
                    np.nan,
                )

                dy = min(
                    diag_h - 1,
                    y // DIAG_Y_FACTOR,
                )

                dx = min(
                    diag_w - 1,
                    x // DIAG_X_FACTOR,
                )

                diagnostic_valid[dy, dx] = True

                diagnostic_probability_data[
                    "rain"
                ][dy, dx] = rain_probability[y, x]

                diagnostic_probability_data[
                    "snow"
                ][dy, dx] = snow_probability[y, x]

                diagnostic_probability_data[
                    "sleet"
                ][dy, dx] = sleet_probability[y, x]

                diagnostic_probability_data[
                    "freezing_rain"
                ][dy, dx] = freezing_rain_probability[y, x]

                diagnostic_probability_data[
                    "melting_energy"
                ][dy, dx] = melting_energy[y, x]

                diagnostic_probability_data[
                    "refreezing_energy"
                ][dy, dx] = refreezing_energy[y, x]

                diagnostic_probability_data[
                    "prob_ice"
                ][dy, dx] = prob_ice[y, x]

    diagnostic_valid_cells = int(
        np.count_nonzero(diagnostic_valid)
    )

    diagnostics = {
        "profile_valid": True,
        "sample_valid_cells": sample_valid_cells,
        "precip_overlap_cells": precip_overlap_cells,
        "full_profile_cells": full_profile_cells,
        "phase_cells": phase_cells,
        "precip_pixels": int(
            np.count_nonzero(precip_mask)
        ),
        "diagnostic_grid": {
            "height": diag_h,
            "width": diag_w,
            "valid_cells": diagnostic_valid_cells,
        },
    }

    return (
        {
            "phase": phase,
            "rain_probability": rain_probability,
            "snow_probability": snow_probability,
            "sleet_probability": sleet_probability,
            "freezing_rain_probability": freezing_rain_probability,
            "melting_energy": melting_energy,
            "refreezing_energy": refreezing_energy,
            "prob_ice": prob_ice,
        },
        diagnostics,
        diagnostic_probability_data,
    )


def main() -> None:
    print("=" * 72)
    print("WINTERRADAR MRMS PROCESSING")
    print("=" * 72)
    print(f"Version: {MAIN_VERSION}")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    download_all()

    ref_path = _path_for("reflectivity")

    ref, lats, lons = get_values_with_time(
        ref_path
    )

    ref = np.asarray(
        ref,
        dtype=np.float32,
    )

    lats = np.asarray(
        lats,
        dtype=np.float64,
    )

    lons = _normalize_longitudes(
        np.asarray(
            lons,
            dtype=np.float64,
        )
    )

    shape = ref.shape

    print(
        f"Reflectivity grid: {shape}"
    )

    precip_flag = try_load_optional(
        "precip_flag",
        shape,
        np.nan,
    )

    bb_top = try_load_optional(
        "bb_top",
        shape,
        np.nan,
    )

    bb_bottom = try_load_optional(
        "bb_bottom",
        shape,
        np.nan,
    )

    rqi = try_load_optional(
        "rqi",
        shape,
        np.nan,
    )

    wetbulb = try_load_optional(
        "wetbulb",
        shape,
        np.nan,
    )

    metadata_path = OUTPUT_DIR / "mrms_current.json"

    mrms_time_utc = get_mrms_valid_time(
        ref_path
    )

    print(
        "Building scientifically based winter phase mask..."
    )

    phase_status = "fallback_initial"
    phase_error = None
    phase_diagnostics = {}
    phase_probability_data = None

    try:
        if not mrms_time_utc:
            raise RuntimeError(
                "MRMS valid time unavailable; cannot synchronize RAP profile."
            )

        result, phase_diagnostics, phase_probability_data = (
            _profile_phase_result(
                ref,
                lats,
                lons,
                mrms_time_utc,
            )
        )

        # RAP can complete successfully but still return zero usable phase
        # cells if the sampled profile fields do not overlap the precipitating
        # MRMS pixels. Do not allow that to produce an effectively transparent
        # phase product. Fall back to the conservative MRMS classifier while
        # preserving the RAP diagnostics for troubleshooting.
        rap_phase_cells = int(
            phase_diagnostics.get(
                "phase_cells",
                0,
            )
        )

        rap_diag_cells = int(
            phase_diagnostics.get(
                "diagnostic_grid",
                {},
            ).get(
                "valid_cells",
                0,
            )
        )

        precip_pixels = int(
            phase_diagnostics.get(
                "precip_pixels",
                0,
            )
        )

        if precip_pixels > 0 and (
            rap_phase_cells == 0
            or rap_diag_cells == 0
        ):
            reason = (
                f"RAP phase returned no usable cells "
                f"(phase_cells={rap_phase_cells}, "
                f"diagnostic_valid_cells={rap_diag_cells}, "
                f"precip_pixels={precip_pixels})"
            )

            print(
                f"  WARNING: {reason}; "
                "using conservative MRMS phase fallback."
            )

            phase_error = reason

            result = classify_initial(
                reflectivity=ref,
                precip_flag=precip_flag,
                bb_top_m=bb_top,
                bb_bottom_m=bb_bottom,
                wetbulb_c=wetbulb,
                freezing_level_m=None,
                rqi=rqi,
            )

            phase_status = (
                "fallback_initial_no_valid_rap_phase"
            )

            diag_h = max(
                1,
                ref.shape[0] // DIAG_Y_FACTOR,
            )

            diag_w = max(
                1,
                ref.shape[1] // DIAG_X_FACTOR,
            )

            phase_probability_data = {
                "rain": np.full(
                    (diag_h, diag_w),
                    np.nan,
                    dtype=np.float32,
                ),
                "snow": np.full(
                    (diag_h, diag_w),
                    np.nan,
                    dtype=np.float32,
                ),
                "sleet": np.full(
                    (diag_h, diag_w),
                    np.nan,
                    dtype=np.float32,
                ),
                "freezing_rain": np.full(
                    (diag_h, diag_w),
                    np.nan,
                    dtype=np.float32,
                ),
                "melting_energy": np.full(
                    (diag_h, diag_w),
                    np.nan,
                    dtype=np.float32,
                ),
                "refreezing_energy": np.full(
                    (diag_h, diag_w),
                    np.nan,
                    dtype=np.float32,
                ),
                "prob_ice": np.full(
                    (diag_h, diag_w),
                    np.nan,
                    dtype=np.float32,
                ),
            }

        else:
            phase_status = "modified_bourgouin_rap"

    except Exception as exc:
        phase_error = (
            f"{type(exc).__name__}: {exc}"
        )

        print(
            "  WARNING: profile phase engine failed; "
            f"retaining conservative fallback: {phase_error}"
        )

        result = classify_initial(
            reflectivity=ref,
            precip_flag=precip_flag,
            bb_top_m=bb_top,
            bb_bottom_m=bb_bottom,
            wetbulb_c=wetbulb,
            freezing_level_m=None,
            rqi=rqi,
        )

        diag_h = max(
            1,
            ref.shape[0] // DIAG_Y_FACTOR,
        )

        diag_w = max(
            1,
            ref.shape[1] // DIAG_X_FACTOR,
        )

        phase_probability_data = {
            "rain": np.full(
                (diag_h, diag_w),
                np.nan,
                dtype=np.float32,
            ),
            "snow": np.full(
                (diag_h, diag_w),
                np.nan,
                dtype=np.float32,
            ),
            "sleet": np.full(
                (diag_h, diag_w),
                np.nan,
                dtype=np.float32,
            ),
            "freezing_rain": np.full(
                (diag_h, diag_w),
                np.nan,
                dtype=np.float32,
            ),
            "melting_energy": np.full(
                (diag_h, diag_w),
                np.nan,
                dtype=np.float32,
            ),
            "refreezing_energy": np.full(
                (diag_h, diag_w),
                np.nan,
                dtype=np.float32,
            ),
            "prob_ice": np.full(
                (diag_h, diag_w),
                np.nan,
                dtype=np.float32,
            ),
        }

    phase_rgba = result_to_phase_rgba(
        result,
        shape,
    )

    precip_type_rgba = result_to_precip_type_rgba(
        result,
        shape,
    )

    radar_rgba = np.zeros(
        (
            shape[0],
            shape[1],
            4,
        ),
        dtype=np.uint8,
    )

    valid_ref = np.isfinite(ref)

    radar_rgba[..., 0] = 0
    radar_rgba[..., 1] = 0
    radar_rgba[..., 2] = 0
    radar_rgba[..., 3] = 0

    levels = np.array(
        [
            0,
            5,
            10,
            15,
            20,
            25,
            34.5,
            35,
            40,
            45,
            50,
            57.5,
            62.5,
            67.5,
            72.5,
            77.5,
            82.5,
            95,
        ],
        dtype=np.float32,
    )

    colors = np.array(
        [
            [50, 50, 50],
            [14, 14, 90],
            [3, 79, 140],
            [7, 162, 182],
            [17, 229, 31],
            [12, 169, 20],
            [6, 104, 8],
            [255, 255, 0],
            [255, 194, 0],
            [255, 140, 0],
            [221, 0, 0],
            [107, 0, 0],
            [255, 163, 255],
            [238, 29, 244],
            [117, 0, 235],
            [0, 255, 219],
            [0, 76, 74],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )

    valid = (
        valid_ref
        & (ref >= 5.0)
    )

    indices = np.searchsorted(
        levels,
        np.where(
            valid,
            ref,
            levels[0],
        ),
        side="right",
    ) - 1

    indices = np.clip(
        indices,
        0,
        len(levels) - 1,
    )

    radar_rgba[..., :3] = colors[
        indices
    ]

    radar_rgba[..., 3] = np.where(
        valid,
        255,
        0,
    ).astype(np.uint8)

    save_rgba_png(
        radar_rgba,
        OUTPUT_DIR / "mrms_current.png",
    )

    save_rgba_png(
        phase_rgba,
        OUTPUT_DIR / "winter_phase_mask.png",
    )

    save_rgba_png(
        precip_type_rgba,
        OUTPUT_DIR / "winter_precip_type.png",
    )

    if phase_probability_data is not None:
        probability_grid = (
            _build_diagnostic_probability_grid(
                phase_probability_data,
                shape,
            )
        )

        np.savez_compressed(
            OUTPUT_DIR / "phase_probabilities.npz",
            **probability_grid,
        )

    diagnostics = dict(
        phase_diagnostics
    )

    diagnostics[
        "phase_status"
    ] = phase_status

    diagnostics[
        "phase_error"
    ] = phase_error

    diagnostics[
        "main_version"
    ] = MAIN_VERSION

    diagnostics[
        "mrms_time"
    ] = (
        mrms_time_utc.isoformat()
        if mrms_time_utc
        else None
    )

    diagnostics_path = (
        OUTPUT_DIR / "mrms_diagnostics.json"
    )

    diagnostics_path.write_text(
        json.dumps(
            diagnostics,
            indent=2,
            default=str,
        )
    )

    metadata = {
        "grid_shape": [
            int(shape[0]),
            int(shape[1]),
        ],
        "bounds": [
            float(np.nanmin(lats)),
            float(np.nanmin(lons)),
            float(np.nanmax(lats)),
            float(np.nanmax(lons)),
        ],
        "projection": (
            "EPSG:4326-native-latlon"
        ),
        "mrms_time": (
            mrms_time_utc.isoformat()
            if mrms_time_utc
            else None
        ),
        "phase_status": phase_status,
        "phase_error": phase_error,
        "main_version": MAIN_VERSION,
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )

    print()
    print("=" * 72)
    print("CORE MRMS OUTPUTS READY")
    print("=" * 72)
    print(
        "  mrms_current.png"
    )
    print(
        "  winter_phase_mask.png"
    )
    print(
        "  winter_precip_type.png"
    )
    print(
        "  mrms_current.json"
    )

    print()
    print(
        f"  Phase status: {phase_status}"
    )

    if phase_error:
        print(
            f"  Phase note: {phase_error}"
        )

    print(
        f"  RAP sample valid cells: "
        f"{phase_diagnostics.get('sample_valid_cells', 0)}"
    )

    print(
        f"  RAP precip overlap cells: "
        f"{phase_diagnostics.get('precip_overlap_cells', 0)}"
    )

    print(
        f"  RAP full profile cells: "
        f"{phase_diagnostics.get('full_profile_cells', 0)}"
    )

    print(
        f"  RAP phase cells: "
        f"{phase_diagnostics.get('phase_cells', 0)}"
    )

    print(
        f"  MRMS precip pixels: "
        f"{phase_diagnostics.get('precip_pixels', 0)}"
    )

    print()
    print(
        "Radar image: "
        f"{radar_rgba.shape[1]}x"
        f"{radar_rgba.shape[0]} RGBA"
    )

    print(
        "Phase image: "
        f"{phase_rgba.shape[1]}x"
        f"{phase_rgba.shape[0]} RGBA"
    )

    print(
        "Grid: "
        f"{metadata['grid_shape']}"
    )

    print(
        "Bounds: "
        f"{metadata['bounds']}"
    )

    print(
        "Projection: "
        f"{metadata['projection']}"
    )

    print(
        "MRMS time: "
        f"{metadata['mrms_time']}"
    )


if __name__ == "__main__":
    main()
