from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

MRMS_BASE = "https://mrms.ncep.noaa.gov/2D"

# Operational MRMS CONUS 2-D fields used by the WinterRadar project.
PRODUCTS = {
    "reflectivity": "ReflectivityAtLowestAltitude",
    "reflectivity_0c": "Reflectivity_0C",
    "precip_flag": "PrecipFlag",
    "precip_rate": "PrecipRate",
    "rhohv": "MergedRhoHV",
    "zdr": "MergedZdr",
    "bb_top": "BrightBandTopHeight",
    "bb_bottom": "BrightBandBottomHeight",
    "rqi": "RadarQualityIndex",
    "surface_temp": "Model_SurfaceTemp",
    "wetbulb": "Model_WetBulbTemp",
    "freezing_level": "Model_0degC_Height",
}

# Product-specific units and MRMS sentinel values.
FIELD_INFO = {
    "ReflectivityAtLowestAltitude": {
        "units": "dBZ",
        "missing": (-99.0,),
        "no_coverage": (-999.0,),
    },
    "PrecipFlag": {
        "units": "flag",
        "missing": (-1.0,),
        "no_coverage": (-3.0,),
    },
    "PrecipRate": {
        "units": "mm/hr",
        "missing": (-1.0,),
        "no_coverage": (-3.0,),
    },
    "MergedRhoHV": {
        "units": "non-dim",
        "missing": (-99.0,),
        "no_coverage": (-999.0,),
    },
    "MergedZdr": {
        "units": "dB",
        "missing": (-99.0,),
        "no_coverage": (-999.0,),
    },
    "BrightBandTopHeight": {
        "units": "m AGL",
        "missing": (-1.0,),
        "no_coverage": (-3.0,),
    },
    "BrightBandBottomHeight": {
        "units": "m AGL",
        "missing": (-1.0,),
        "no_coverage": (-3.0,),
    },
    "RadarQualityIndex": {
        "units": "non-dim",
        "missing": (-1.0,),
        "no_coverage": (-3.0,),
    },
    "Model_SurfaceTemp": {
        "units": "C",
        "missing": (-99.0,),
        "no_coverage": (-999.0,),
    },
    "Model_WetBulbTemp": {
        "units": "C",
        "missing": (-99.0,),
        "no_coverage": (-999.0,),
    },
    "Model_0degC_Height": {
        "units": "m MSL",
        "missing": (-1.0,),
        "no_coverage": (-3.0,),
    },
}

MISSING = -999.0


def field_info(product: str) -> dict:
    """Return metadata and sentinel values for an MRMS product."""
    return FIELD_INFO.get(
        product,
        {
            "units": "unknown",
            "missing": (),
            "no_coverage": (),
        },
    )
# ------------------------------------------------------------
# Vertical dual-polarization MRMS CAPPIs
# ------------------------------------------------------------

VERTICAL_DUALPOL_LEVELS_KM = [
    0.50,
    1.00,
    1.50,
    2.00,
    2.50,
    3.00,
    3.50,
    4.00,
]

VERTICAL_DUALPOL_PRODUCTS = {
    "rhohv": "MergedRhoHV",
    "zdr": "MergedZdr",
}
