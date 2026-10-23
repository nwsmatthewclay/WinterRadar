# Winter Radar Composite

The live WinterRadar composite is a display renderer built from the existing scientific products.

## Encoding

- **Hue / color family:** precipitation type from the existing phase classifier.
- **Brightness / intensity:** native MRMS MergedReflectivityQCComposite reflectivity in dBZ.
- **Saturation / opacity:** phase-classification confidence from 0 to 1.
- **Rain:** retains the MRMS BR HiRes reflectivity palette.
- **Snow:** blue family.
- **Sleet:** purple family.
- **Freezing rain:** red/orange ice family.
- **Mixed:** magenta/purple family.
- **Uncertain:** neutral gray family.

The renderer does not create a new precipitation-type classification and does not replace the underlying RAP/Modified-Bourgouin or conservative MRMS fallback solution.

## Browser assets

The native raster is written as `outputs/winter_radar_composite.png`. The Web Mercator browser copy is written as `outputs/winter_radar_composite_web.png` and a lossless WebP companion.

`winter_precip_type.*` remains as a compatibility alias for the existing archive/history pipeline.

## Design intent

The interactive map is styled as a dark meteorological workstation. The default layer is the Winter Radar Composite; raw MRMS, Winter Phase Mask, radar phase evidence, and phase agreement remain available as separate analysis layers.


## Regional display and interpolation

The browser-facing operational layers use an expanded Northeast domain
(35.0–50.5°N, 88.0–63.0°W). The continuous MRMS reflectivity display uses
gentle bilinear upsampling for presentation only; native scientific values
and categorical winter-phase masks are not smoothed. Sparse radar phase evidence
and phase-agreement overlays use per-class alpha interpolation when expanded
from the reduced diagnostic grid. This anti-aliases square edges while retaining
fixed class colors and leaving the scientific category arrays unchanged. Radar
phase evidence and phase agreement have regional browser copies so their controls
do not need to fetch full national diagnostic images.
