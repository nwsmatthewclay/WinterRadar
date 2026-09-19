#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-outputs}"
RADAR="$ROOT/mrms_current.png"
PHASE="$ROOT/winter_phase_mask.png"
META="$ROOT/mrms_current.json"

printf 'Verifying core MRMS outputs in %s\n' "$ROOT"

for file in "$RADAR" "$PHASE" "$META"; do
  if [[ ! -s "$file" ]]; then
    echo "ERROR: missing or empty: $file" >&2
    exit 1
  fi
  printf '  OK: %s (%s bytes)\n' "$file" "$(stat -c '%s' "$file")"
done

python - "$ROOT" <<'PY2'
from pathlib import Path
import json
import sys
from PIL import Image

root = Path(sys.argv[1])
radar = root / "mrms_current.png"
phase = root / "winter_phase_mask.png"
meta = root / "mrms_current.json"

with Image.open(radar) as im:
    print(f"  Radar image: {im.size[0]}x{im.size[1]} {im.mode}")
    if im.mode != "RGBA":
        raise SystemExit("ERROR: radar image is not RGBA")
    if im.size != (7000, 3500):
        raise SystemExit(f"ERROR: unexpected radar dimensions: {im.size}")

with Image.open(phase) as im:
    print(f"  Phase image: {im.size[0]}x{im.size[1]} {im.mode}")
    if im.mode != "RGBA":
        raise SystemExit("ERROR: phase image is not RGBA")
    if im.size != (7000, 3500):
        raise SystemExit(f"ERROR: unexpected phase dimensions: {im.size}")

with meta.open(encoding="utf-8") as fh:
    data = json.load(fh)

shape = data.get("grid_shape")
if shape != [3500, 7000]:
    raise SystemExit(f"ERROR: unexpected grid_shape: {shape}")

bounds = data.get("bounds")
if not isinstance(bounds, list) or len(bounds) != 4:
    raise SystemExit(f"ERROR: invalid bounds: {bounds}")
south, west, north, east = map(float, bounds)
if not (20.0 <= south < north <= 55.0):
    raise SystemExit(f"ERROR: invalid latitude bounds: {bounds}")
if not (-130.0 <= west < east <= -60.0):
    raise SystemExit(f"ERROR: invalid longitude bounds: {bounds}")

print(f"  Grid: {shape}")
print(f"  Bounds: {bounds}")
print(f"  Projection: {data.get('projection')}")
print(f"  MRMS time: {data.get('mrms_time_utc', 'not available')}")
print("CORE MRMS OUTPUT QC PASSED")
PY2
