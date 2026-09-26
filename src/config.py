from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

# Official operational MRMS 2-D archive used by the project.
MRMS_BASE = "https://mrms.ncep.noaa.gov/2D"

# Keep the complete product catalog for compatibility with the diagnostic
# scripts already in the repository.  The live map does NOT download all of
# these products; see LIVE_PRODUCTS and OPTIONAL_DIAGNOSTIC_PRODUCTS below.
PRODUCTS = {
    "reflectivity": "MergedReflectivityComposite",
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

# Critical live-map fields.  Reflectivity is the only hard requirement for
# the radar image itself.  The remaining fields improve the winter-phase mask
# and are downloaded on a best-effort basis so a single upstream issue cannot
# take the live radar completely offline.
LIVE_PRODUCTS = {
    "reflectivity": PRODUCTS["reflectivity"],
    "precip_flag": PRODUCTS["precip_flag"],
    "bb_top": PRODUCTS["bb_top"],
    "bb_bottom": PRODUCTS["bb_bottom"],
    "rqi": PRODUCTS["rqi"],
    "wetbulb": PRODUCTS["wetbulb"],
}

REQUIRED_LIVE_PRODUCTS = {
    "reflectivity": PRODUCTS["reflectivity"],
}

# These products are for diagnostics/research only. They are intentionally
# outside the live MRMS critical path.
OPTIONAL_DIAGNOSTIC_PRODUCTS = {
    "precip_rate": PRODUCTS["precip_rate"],
    "rhohv": PRODUCTS["rhohv"],
    "zdr": PRODUCTS["zdr"],
    "surface_temp": PRODUCTS["surface_temp"],
    "freezing_level": PRODUCTS["freezing_level"],
    "reflectivity_0c": "Reflectivity_0C",
}

# Product-specific units and MRMS sentinel values.
FIELD_INFO = {
    "MergedReflectivityComposite": {
        "units": "dBZ",
        "missing": (-99.0,),
        "no_coverage": (-999.0,),
    },
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
    "Reflectivity_0C": {
        "units": "dBZ",
        "missing": (-99.0,),
        "no_coverage": (-999.0,),
    },
}

# Network timing. Required radar gets more time/retries; optional phase fields
# are deliberately less persistent so they cannot hold up the live map.
REQUIRED_CONNECT_TIMEOUT = 20
REQUIRED_READ_TIMEOUT = 120
REQUIRED_ATTEMPTS = 4
OPTIONAL_CONNECT_TIMEOUT = 15
OPTIONAL_READ_TIMEOUT = 45
OPTIONAL_ATTEMPTS = 2

MISSING = -999.0


def field_info(product: str) -> dict:
    return FIELD_INFO.get(
        product,
        {
            "units": "unknown",
            "missing": (),
            "no_coverage": (),
        },
    )


# Vertical dual-polarization MRMS CAPPI products.
VERTICAL_DUALPOL_LEVELS_KM = [0.50, 1.00, 1.50, 2.00, 2.50, 3.00, 3.50, 4.00]
VERTICAL_DUALPOL_PRODUCTS = {"rhohv": "MergedRhoHV", "zdr": "MergedZdr"}
MRMS_3D_RHOHV_BASE = "https://mrms.ncep.noaa.gov/3DRhoHV"
MRMS_3D_ZDR_BASE = "https://mrms.ncep.noaa.gov/3DZdr"
