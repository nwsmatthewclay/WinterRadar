# MRMS Winter Radar Viewer

Initial prototype for a CONUS MRMS winter precipitation viewer and social-media graphic generator.

## Architecture

- `src/` — Python MRMS download, processing, classification, and rendering
- `web/` — static interactive Leaflet viewer
- `outputs/` — generated PNG/JSON assets published by GitHub Pages
- `.github/workflows/` — scheduled GitHub Actions workflow

## Initial MRMS fields

- ReflectivityAtLowestAltitude
- PrecipFlag
- BrightBandTopHeight
- BrightBandBottomHeight
- RadarQualityIndex
- Model_WetBulbTemp
- Model_0degC_Height

The phase classifier is intentionally modular. The first version is radar-assisted and conservative; a later module will add an HRRR/RAP vertical wet-bulb profile and revised Bourgouin-style energy calculation for snow/sleet/freezing-rain separation.

## GitHub Pages

Set GitHub Pages to use **GitHub Actions** as the deployment source. The workflow builds a `site/` artifact containing the viewer and generated MRMS image, then deploys that artifact to Pages. GitHub's Pages workflow supports this artifact-based deployment model.

The scheduled workflow runs every 5 minutes. GitHub Actions supports scheduled workflows as frequently as every 5 minutes, although scheduled jobs can occasionally be delayed under high load.
