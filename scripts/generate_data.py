import requests
import numpy as np
import geopandas as gpd
from shapely import contains_xy
from scipy.interpolate import griddata
import json, time
from datetime import datetime, timezone
from pathlib import Path

# ---------- CONFIG ----------
# Paths are anchored to the repo root so the script works from any cwd.
REPO_ROOT    = Path(__file__).resolve().parent.parent
GEOJSON_FILE = REPO_ROOT / 'neighborhoods.geojson'
OUTPUT_JSON  = REPO_ROOT / 'public' / 'data.json'
GRID_SIZE    = 100
PADDING      = 1.15   # 15% padding around SF

# ---------- DATA SOURCE: Open-Meteo ----------
# Open-Meteo returns gridded model surface temperature corrected to the 90 m DEM
# elevation of each requested point. Measured across SF: corr(elevation, temp)
# = -0.90 — Twin Peaks (266 m) reads ~3.5 F colder than sea level. Dense point
# sampling therefore captures real topography-driven microclimates.
OM_MODEL       = 'gfs_seamless'   # NOAA GFS + HRRR blend (best_match is identical here)
OM_BATCH       = 100              # coordinates per request
OM_SAMPLE_GRID = 20               # 20x20 = 400 sample points (~800 m spacing)
MIN_SAMPLE_FRAC = 0.5             # abort if <50% of sample points return data

# ---------- Load geometry ----------
gdf = gpd.read_file(GEOJSON_FILE)
gdf = gdf.set_crs(epsg=4326) if gdf.crs is None else gdf.to_crs(epsg=4326)
name_col = next((c for c in ['canonicalName','name','neighborhood','nhood','NEIGHBORHOOD']
                 if c in gdf.columns), None)
if name_col is None:
    raise ValueError(f"No name column. Columns: {list(gdf.columns)}")

pts = gdf.geometry.representative_point()
gdf['lon'], gdf['lat'] = pts.x, pts.y
print(f"Loaded {len(gdf)} neighborhoods (name: '{name_col}')")

# ---------- Sampling ----------
# (runs after the grid extent is computed below)
def fetch_openmeteo(coords):
    """coords: list of (lat, lon). Returns list of dicts (lat, lon, elev, temp_f)."""
    out = []
    for k in range(0, len(coords), OM_BATCH):
        chunk = coords[k:k + OM_BATCH]
        params = {
            "latitude":  ",".join(f"{la:.5f}" for la, _ in chunk),
            "longitude": ",".join(f"{lo:.5f}" for _, lo in chunk),
            "current": "temperature_2m",
            "temperature_unit": "fahrenheit",
            "models": OM_MODEL,
            "timezone": "UTC",
        }
        for attempt in range(4):
            try:
                r = requests.get("https://api.open-meteo.com/v1/forecast",
                                 params=params, timeout=60)
                d = r.json()
                if isinstance(d, dict):          # API-level error (esp. rate limit)
                    print(f"  ! {d.get('reason', d)}")
                    time.sleep(20)
                    continue
                for x in d:
                    out.append({
                        "lat": x["latitude"],
                        "lon": x["longitude"],
                        "elev": x.get("elevation"),
                        "temp_f": x["current"]["temperature_2m"],
                    })
                break
            except Exception as e:
                print(f"  ! request failed ({e}), retrying...")
                time.sleep(5)
        time.sleep(1.5)   # stay under Open-Meteo's ~600 calls/min limit
    return out


# ---------- Square grid ----------
sf_minx, sf_miny, sf_maxx, sf_maxy = gdf.total_bounds
cx = (sf_minx + sf_maxx) / 2
cy = (sf_miny + sf_maxy) / 2
mid_lat = cy
w_km = (sf_maxx - sf_minx) * 111.32 * np.cos(np.radians(mid_lat))
h_km = (sf_maxy - sf_miny) * 110.57
side_km = max(w_km, h_km) * PADDING
half_w = (side_km / 2) / (111.32 * np.cos(np.radians(mid_lat)))
half_h = (side_km / 2) / 110.57
minx, maxx = cx - half_w, cx + half_w
miny, maxy = cy - half_h, cy + half_h

xs = np.linspace(minx, maxx, GRID_SIZE)
ys = np.linspace(maxy, miny, GRID_SIZE)
grid_x, grid_y = np.meshgrid(xs, ys)
print(f"\nGrid: {GRID_SIZE}×{GRID_SIZE} (square, {side_km:.1f} km side)")

# ---------- Sample Open-Meteo ----------
sample_xs = np.linspace(minx, maxx, OM_SAMPLE_GRID)
sample_ys = np.linspace(miny, maxy, OM_SAMPLE_GRID)
coords = [(la, lo) for la in sample_ys for lo in sample_xs]
spacing_m = side_km / max(1, OM_SAMPLE_GRID - 1) * 1000
print(f"\nSampling Open-Meteo ({OM_MODEL}): {len(coords)} points "
      f"(~{spacing_m:.0f} m spacing)...")

samples = [s for s in fetch_openmeteo(coords) if s["temp_f"] is not None]
print(f"  received {len(samples)}/{len(coords)} points")

if len(samples) < MIN_SAMPLE_FRAC * len(coords):
    raise SystemExit(
        f"\n✗ Only {len(samples)}/{len(coords)} sample points returned data.\n"
        f"  Open-Meteo may be rate-limited or down — refusing to overwrite "
        f"{OUTPUT_JSON.name}.\n  Existing data was left untouched.")

elevs = [s["elev"] for s in samples if s["elev"] is not None]
if elevs:
    print(f"  elevation range: {min(elevs):.0f}–{max(elevs):.0f} m")

# ---------- Interpolate & mask ----------
points = np.column_stack([[s["lon"] for s in samples],
                          [s["lat"] for s in samples]])
values = np.array([s["temp_f"] for s in samples])
lin = griddata(points, values, (grid_x, grid_y), method='linear')
nn  = griddata(points, values, (grid_x, grid_y), method='nearest')
interp = np.where(np.isnan(lin), nn, lin)

inside = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
for geom in gdf.geometry:
    inside |= contains_xy(geom, grid_x, grid_y)
print(f"Cells inside SF: {inside.sum()} / {GRID_SIZE*GRID_SIZE}")

# ---------- Labels ----------
labels = []
for _, row in gdf.iterrows():
    i = int(np.argmin(np.abs(xs - row['lon'])))
    j = int(np.argmin(np.abs(ys - row['lat'])))
    if inside[j, i] and not np.isnan(interp[j, i]):
        labels.append({
            "r": j, "c": i, "name": row[name_col],
            "temp": round(float(interp[j, i]), 1),
        })

# ---------- Build JSON ----------
temps_flat = []
for j in range(GRID_SIZE):
    for i in range(GRID_SIZE):
        if inside[j, i] and not np.isnan(interp[j, i]):
            temps_flat.append(round(float(interp[j, i]), 1))
        else:
            temps_flat.append(None)

valid = [t for t in temps_flat if t is not None]
if not valid:
    raise SystemExit(f"✗ No valid cells produced — refusing to overwrite {OUTPUT_JSON.name}.")

# minTemp/maxTemp report the range observed in THIS build, for display only.
# The page's colour scale is fixed at 27-106 F (SF's all-time records), so a
# given colour always maps to the same temperature.
data_lo, data_hi = min(valid), max(valid)

output = {
    "generated": datetime.now(timezone.utc).isoformat(),
    "cols": GRID_SIZE,
    "rows": GRID_SIZE,
    "minTemp": round(data_lo, 1),
    "maxTemp": round(data_hi, 1),
    "temps": temps_flat,
    "labels": labels,
    "sources": {
        "provider": "Open-Meteo",
        "model": OM_MODEL,
        "samplePoints": len(samples),
        "spacingM": round(spacing_m),
        "elevationMin": round(min(elevs)) if elevs else None,
        "elevationMax": round(max(elevs)) if elevs else None,
    },
}
with open(OUTPUT_JSON, 'w') as f:
    json.dump(output, f, separators=(',', ':'))

print(f"\n✅ Wrote {OUTPUT_JSON}")
print(f"   {len(valid)} cells, {len(labels)} labels")
print(f"   Observed range: {data_lo:.1f}°F – {data_hi:.1f}°F "
      f"(fixed colour scale 27–106°F)")
print(f"   Source: Open-Meteo ({OM_MODEL}), {len(samples)} sample points")