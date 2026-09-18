from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

MRMS_BASE = "https://mrms.ncep.noaa.gov/2D"

PRODUCTS = {
    "reflectivity": "ReflectivityAtLowestAltitude",
    "precip_flag": "PrecipFlag",

    # Dual-polarization fields
    "rhohv": "MergedRhoHV",
    "zdr": "MergedZdr",

    # Melting-layer information
    "bb_top": "BrightBandTopHeight",
    "bb_bottom": "BrightBandBottomHeight",

    # Radar quality
    "rqi": "RadarQualityIndex",

    # 2-D environmental fields
    "surface_temp": "Model_SurfaceTemp",
    "wetbulb": "Model_WetBulbTemp",
    "freezing_level": "Model_0degC_Height",
}

# MRMS grid approximately covers the CONUS on a 0.01-degree grid.
# We retain the native grid in the first prototype.
MISSING = -999.0
