from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from classifier import (
    CLEAR,
    FZRA,
    MIXED,
    RAIN,
    SLEET,
    SNOW,
    UNKNOWN,
    ClassificationResult,
)


# ---------------------------------------------------------------------
# Winter mask colors
#
# These are intentionally semi-transparent so the radar underneath
# remains visible.
# ---------------------------------------------------------------------

PHASE_COLORS = {
    SNOW: (55, 145, 255, 105),
    SLEET: (185, 80, 220, 115),
    FZRA: (225, 55, 70, 115),
    MIXED: (220, 95, 195, 105),
}


# Unknown is intentionally more subdued.
UNKNOWN_COLOR = (150, 150, 150, 75)


def result_to_rgba(
    result: ClassificationResult,
) -> np.ndarray:

    h, w = result.phase.shape

    # Start completely transparent.
    rgba = np.zeros(
        (h, w, 4),
        dtype=np.uint8,
    )

    # ---------------------------------------------------------------
    # Winter phases
    # ---------------------------------------------------------------

    for phase, color in PHASE_COLORS.items():

        mask = (
            result.phase == phase
        )

        rgba[mask] = color

    # ---------------------------------------------------------------
    # Unknown / uncertain winter precipitation
    # ---------------------------------------------------------------

    unknown = (
        result.phase == UNKNOWN
    )

    rgba[unknown] = UNKNOWN_COLOR

    # ---------------------------------------------------------------
    # CLEAR and RAIN intentionally remain:
    #
    # (0, 0, 0, 0)
    #
    # This is the important behavior.
    # ---------------------------------------------------------------

    return rgba


def save_rgba_png(
    rgba: np.ndarray,
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.fromarray(
        rgba,
        mode="RGBA",
    ).save(
        path,
        optimize=True,
    )


def write_metadata(
    result: ClassificationResult,
    path: Path,
) -> None:

    unique, counts = np.unique(
        result.phase,
        return_counts=True,
    )

    counts_by_phase = {
        str(int(k)): int(v)
        for k, v in zip(unique, counts)
    }

    total = int(
        result.phase.size
    )

    phase_names = {
        CLEAR: "clear",
        RAIN: "rain",
        SNOW: "snow",
        SLEET: "sleet",
        FZRA: "freezing_rain",
        MIXED: "mixed",
        UNKNOWN: "unknown",
    }

    counts_named = {
        phase_names.get(
            int(k),
            f"phase_{int(k)}",
        ): int(v)
        for k, v in zip(unique, counts)
    }

    metadata = {
        "counts_by_phase": counts_by_phase,
        "counts_by_phase_name": counts_named,
        "total_pixels": total,
        "winter_pixels": int(
            np.count_nonzero(
                np.isin(
                    result.phase,
                    [
                        SNOW,
                        SLEET,
                        FZRA,
                        MIXED,
                        UNKNOWN,
                    ],
                )
            )
        ),
        "transparent_pixels": int(
            np.count_nonzero(
                np.isin(
                    result.phase,
                    [
                        CLEAR,
                        RAIN,
                    ],
                )
            )
        ),
    }

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )
