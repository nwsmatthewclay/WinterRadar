from __future__ import annotations

"""Build compact QC products for the HRRR/Modified-Bourgouin phase engine.

Inputs are written by src/main.py:
  outputs/phase_probabilities.npz
  outputs/mrms_current.json

Outputs:
  outputs/phase_diagnostics.png   -- six-panel probability/energy QC image
  outputs/phase_probability.png   -- dominant-phase probability layer
  outputs/phase_qc.json           -- machine-readable summary
"""

import json
from pathlib import Path

import numpy as np
import requests
from matplotlib.path import Path as MplPath
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

COLORS = {
    "rain": (80, 150, 220),
    "snow": (55, 145, 255),
    "sleet": (185, 80, 220),
    "freezing_rain": (225, 55, 70),
    "mixed": (220, 95, 195),
    "unknown": (145, 150, 155),
}


BTV_CWA_BOUNDS = [42.55, -75.30, 45.20, -70.75]


def _load_btv_cwa_geometry() -> list[np.ndarray]:
    """Fetch the current NWS BTV CWA polygon used to mask probability graphics."""
    url = (
        "https://mapservices.weather.noaa.gov/static/rest/services/"
        "nws_reference_maps/nws_reference_map/FeatureServer/1/query"
    )
    params = {
        "where": "cwa='BTV'",
        "outFields": "cwa",
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "geojson",
    }

    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    payload = response.json()

    rings: list[np.ndarray] = []
    for feature in payload.get("features", []):
        geometry = feature.get("geometry") or {}
        coordinates = geometry.get("coordinates", [])
        if geometry.get("type") == "Polygon":
            polygons = [coordinates]
        elif geometry.get("type") == "MultiPolygon":
            polygons = coordinates
        else:
            polygons = []

        for polygon in polygons:
            for ring in polygon[:1]:
                if len(ring) >= 3:
                    rings.append(np.asarray(ring, dtype=np.float64))

    if not rings:
        raise RuntimeError("NWS BTV CWA geometry returned no polygon rings.")

    return rings


def _btv_mask(data_shape: tuple[int, int], meta: dict) -> np.ndarray:
    """Return a boolean mask for the BTV CWA on the diagnostic probability grid."""
    south, west, north, east = (
        float(meta.get("bounds", BTV_CWA_BOUNDS)[0]),
        float(meta.get("bounds", BTV_CWA_BOUNDS)[1]),
        float(meta.get("bounds", BTV_CWA_BOUNDS)[2]),
        float(meta.get("bounds", BTV_CWA_BOUNDS)[3]),
    )

    # The diagnostic arrays are a 10x reduction of the native MRMS lat/lon grid.
    # Their geographic extent is therefore represented by the same outer bounds.
    h, w = data_shape
    lats = np.linspace(north, south, h)
    lons = np.linspace(west, east, w)
    xx, yy = np.meshgrid(lons, lats)

    try:
        rings = _load_btv_cwa_geometry()
        points = np.column_stack((xx.ravel(), yy.ravel()))
        mask = np.zeros(points.shape[0], dtype=bool)
        for ring in rings:
            mask |= MplPath(ring).contains_points(points)
        return mask.reshape(data_shape)
    except Exception as exc:
        # Diagnostics should remain available if the public GIS service has a
        # transient outage. Fall back to the established BTV regional bounds,
        # but make the fallback explicit in the QC metadata.
        print(f"WARNING: unable to load current NWS BTV CWA polygon: {exc}")
        regional = (
            (yy >= BTV_CWA_BOUNDS[0]) &
            (yy <= BTV_CWA_BOUNDS[2]) &
            (xx >= BTV_CWA_BOUNDS[1]) &
            (xx <= BTV_CWA_BOUNDS[3])
        )
        return regional


def font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def normalize_percent(a: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(a, nan=0.0), 0.0, 100.0) / 100.0


def probability_rgb(prob: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    p = normalize_percent(prob)
    rgb = np.empty((*p.shape, 3), dtype=np.uint8)
    base = np.asarray(color, dtype=np.float32)
    # Keep low probabilities visually quiet while preserving the actual value.
    rgb[:] = np.clip(245.0 - (245.0 - base) * p[..., None], 0, 255).astype(np.uint8)
    return rgb


def energy_rgb(a: np.ndarray, vmax: float) -> np.ndarray:
    x = np.clip(np.nan_to_num(a, nan=0.0), 0.0, vmax) / vmax
    # Neutral -> orange/red ramp, intentionally simple and readable.
    r = 245.0 * x + 45.0 * (1 - x)
    g = 245.0 * (1 - x) + 75.0 * x
    b = 245.0 * (1 - x)
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def save_probability_layer(data: dict[str, np.ndarray], path: Path) -> None:
    names = ["rain", "snow", "sleet", "freezing_rain"]
    stack = np.stack([np.nan_to_num(data[n], nan=0.0) for n in names], axis=0)
    top = np.argmax(stack, axis=0)
    maxp = np.max(stack, axis=0)
    colors = np.asarray([
        COLORS["rain"], COLORS["snow"], COLORS["sleet"], COLORS["freezing_rain"]
    ], dtype=np.uint8)
    rgb = colors[top]
    # Alpha is probability, with a modest floor so 50% areas remain visible.
    alpha = np.where(maxp >= 20.0, np.clip(70.0 + maxp * 1.6, 0, 230), 0).astype(np.uint8)
    rgba = np.dstack([rgb, alpha])
    Image.fromarray(rgba, "RGBA").save(path, optimize=True)


def build_panel(title: str, arr: np.ndarray, kind: str, size: tuple[int, int]) -> Image.Image:
    w, h = size
    valid = np.isfinite(arr)

    if kind == "prob":
        color = COLORS[title.lower().replace(" ", "_")]
        unit = "%"
        rgb = probability_rgb(arr, color) if np.any(valid) else np.full((*arr.shape, 3), 232, dtype=np.uint8)
    else:
        unit = "J/kg"
        if np.any(valid):
            vmax = 250.0 if "Melting" in title else 150.0
            rgb = energy_rgb(arr, vmax)
        else:
            rgb = np.full((*arr.shape, 3), 232, dtype=np.uint8)

    # Keep the compact diagnostic grid crisp when enlarged for the dashboard.
    img = Image.fromarray(rgb, "RGB").resize((w, h), Image.Resampling.NEAREST)
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle((0, 0, w, 34), fill=(255, 255, 255, 225))

    if np.any(valid):
        mn = float(np.nanmin(arr))
        mx = float(np.nanmax(arr))
        mean = float(np.nanmean(arr))
        text = f"{title}   mean {mean:.1f}{unit}   max {mx:.1f}{unit}"
    else:
        text = f"{title}   no valid data"
        d.text((w // 2, h // 2), "No valid data", fill=(110, 120, 130, 255),
               font=font(20), anchor="mm")

    d.text((9, 9), text, fill=(25, 35, 45, 255), font=font(14))
    return img


def main() -> None:
    npz = OUT / "phase_probabilities.npz"
    meta_path = OUT / "mrms_current.json"
    if not npz.exists():
        raise FileNotFoundError(npz)

    with np.load(npz) as z:
        data = {k: np.asarray(z[k], dtype=np.float32) for k in z.files}

    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    required = ["rain", "snow", "sleet", "freezing_rain", "melting_energy", "refreezing_energy", "prob_ice"]
    missing = [k for k in required if k not in data]
    if missing:
        raise RuntimeError(f"Missing diagnostic arrays: {missing}")

    # Restrict all probability/energy graphics to the current BTV CWA.
    # The native radar remains full-CONUS; only the diagnostic graphics are
    # geographically masked here.
    btv_mask = _btv_mask(data["rain"].shape, meta)
    rows = np.where(btv_mask.any(axis=1))[0]
    cols = np.where(btv_mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        raise RuntimeError("BTV CWA mask contains no diagnostic grid cells.")

    r0, r1 = int(rows.min()), int(rows.max()) + 1
    c0, c1 = int(cols.min()), int(cols.max()) + 1

    for key in data:
        data[key] = np.where(btv_mask, data[key], np.nan)[r0:r1, c0:c1]

    save_probability_layer(data, OUT / "phase_probability.png")

    panels = [
        ("Rain", data["rain"], "prob"),
        ("Snow", data["snow"], "prob"),
        ("Sleet", data["sleet"], "prob"),
        ("Freezing Rain", data["freezing_rain"], "prob"),
        ("Melting Energy", data["melting_energy"], "energy"),
        ("Refreezing Energy", data["refreezing_energy"], "energy"),
    ]

    pw, ph = 760, 390
    canvas = Image.new("RGB", (pw * 2, ph * 3), (238, 241, 243))
    for i, (title, arr, kind) in enumerate(panels):
        panel = build_panel(title, arr, kind, (pw, ph))
        canvas.paste(panel, ((i % 2) * pw, (i // 2) * ph))

    d = ImageDraw.Draw(canvas, "RGBA")
    d.rectangle((0, canvas.height - 42, canvas.width, canvas.height), fill=(255, 255, 255, 235))
    engine = meta.get("phase_status", "unknown")
    d.text((12, canvas.height - 30), f"Phase engine: {engine}  •  Diagnostic resolution: {data['rain'].shape[1]} × {data['rain'].shape[0]}", fill=(35, 45, 55, 255), font=font(13))
    canvas.save(OUT / "phase_diagnostics.png", optimize=True)

    stack = np.stack([data[n] for n in ("rain", "snow", "sleet", "freezing_rain")], axis=0)
    finite = np.isfinite(stack).any(axis=0)
    top = np.argmax(np.nan_to_num(stack, nan=-1.0), axis=0)
    safe_stack = np.nan_to_num(stack, nan=-1.0)
    maxp = np.max(safe_stack, axis=0)
    second = np.partition(safe_stack, -2, axis=0)[-2]

    qc = {
        "status": "ok",
        "engine": meta.get("phase_diagnostics", {}).get("engine", "unknown"),
        "phase_status": meta.get("phase_status", "unknown"),
        "hrrr_profile_file": meta.get("phase_diagnostics", {}).get("hrrr_profile_file"),
        "pressure_levels_hpa": meta.get("phase_diagnostics", {}).get("pressure_levels_hpa", []),
        "diagnostic_grid": {"width": int(data["rain"].shape[1]), "height": int(data["rain"].shape[0])},
        "mean_probability_percent": {},
        "max_probability_percent": {},
        "dominant_probability_area_percent": {},
        "high_confidence_area_percent": {},
        "mean_melting_energy_jkg": float(np.nanmean(data["melting_energy"])) if np.isfinite(data["melting_energy"]).any() else None,
        "mean_refreezing_energy_jkg": float(np.nanmean(data["refreezing_energy"])) if np.isfinite(data["refreezing_energy"]).any() else None,
        "mean_prob_ice_percent": float(np.nanmean(data["prob_ice"])) if np.isfinite(data["prob_ice"]).any() else None,
        "qc_definition": "High confidence = dominant probability >= 70% and >= 15 percentage points above runner-up.",
        "domain": "WFO BTV County Warning Area",
        "domain_cwa": "BTV",
    }

    names = ["rain", "snow", "sleet", "freezing_rain"]
    for i, name in enumerate(names):
        arr = data[name]
        finite_arr = np.isfinite(arr)
        qc["mean_probability_percent"][name] = float(np.nanmean(arr)) if np.any(finite_arr) else None
        qc["max_probability_percent"][name] = float(np.nanmax(arr)) if np.any(finite_arr) else None
        dom = finite & (top == i)
        hi = dom & (maxp >= 70.0) & ((maxp - second) >= 15.0)
        qc["dominant_probability_area_percent"][name] = float(100.0 * np.count_nonzero(dom) / max(np.count_nonzero(finite), 1))
        qc["high_confidence_area_percent"][name] = float(100.0 * np.count_nonzero(hi) / max(np.count_nonzero(finite), 1))

    (OUT / "phase_qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print("PHASE DIAGNOSTICS READY")
    print(f"  {OUT / 'phase_diagnostics.png'}")
    print(f"  {OUT / 'phase_probability.png'}")
    print(f"  {OUT / 'phase_qc.json'}")


if __name__ == "__main__":
    main()
