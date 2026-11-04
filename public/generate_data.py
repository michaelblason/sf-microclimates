import requests
import numpy as np
import geopandas as gpd
from shapely import contains_xy
from scipy.interpolate import griddata
import json, difflib, re
from datetime import datetime, timezone

# ---------- CONFIG ----------
GEOJSON_FILE = 'neighborhoods.geojson'
OUTPUT_JSON  = 'docs/data.json'
GRID_SIZE    = 100
MIN_TEMP, MAX_TEMP = 50, 70
DEFAULT_TEMP = 55.0
PADDING      = 1.15   # 15% padding around SF

NAME_OVERRIDES = {
    "bayview hunters point": "bayview",
    "castro/upper market":   "castro",
    "financial district/south beach": "financial_district",
    "sunset/parkside":       "parkside",
    "seacliff":              "sea_cliff",
    "haight ashbury":        "haight",
}

def normalize(s):
    s = s.lower().strip().replace('&', ' and ')
    s = re.sub(r'[^a-z0-9 ]+', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

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

# ---------- API keys ----------
print("\nFetching Microclimates API keys...")
try:
    r = requests.get("https://microclimates.solofounders.com/neighborhoods", timeout=10)
    api_list = r.json().get("neighborhoods", [])
    api_map = {normalize(i["name"]): i["key"] for i in api_list}
    print(f"  {len(api_map)} keys")
except Exception as e:
    print(f"  Failed: {e}")
    api_map = {}

def best_key(name):
    n = normalize(name)
    if n in NAME_OVERRIDES:  return NAME_OVERRIDES[n]
    if n in api_map:         return api_map[n]
    m = difflib.get_close_matches(n, api_map.keys(), n=1, cutoff=0.6)
    return api_map[m[0]] if m else None

def fetch_mc(key):
    try:
        r = requests.get(f"https://microclimates.solofounders.com/sf-weather/{key}", timeout=5)
        if r.ok:
            t = r.json().get('temp_f')
            return float(t) if t is not None else None
    except Exception: pass
    return None

def fetch_om(lat, lon):
    try:
        r = requests.get(
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m&temperature_unit=fahrenheit"
            f"&timezone=America/Los_Angeles", timeout=8)
        if r.ok:
            return float(r.json()["current"]["temperature_2m"])
    except Exception: pass
    return None

# ---------- Fetch ----------
print("\nFetching temps...")
sources, temps_list = [], []
for _, row in gdf.iterrows():
    key = best_key(row[name_col])
    t, src = None, ""
    if key:
        t = fetch_mc(key)
        if t is not None: src = "mc"
    if t is None:
        t = fetch_om(row['lat'], row['lon'])
        if t is not None: src = "om"
    if t is None:
        t, src = DEFAULT_TEMP, "default"
    print(f"  {row[name_col]:<32} {t:>5.1f}°F ({src})")
    temps_list.append(t); sources.append(src)
gdf['temp_f'] = temps_list

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

# ---------- Interpolate & mask ----------
points = np.column_stack([gdf.lon, gdf.lat])
values = gdf.temp_f.values
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
    if inside[j, i]:
        labels.append({"r": j, "c": i, "name": row[name_col]})

# ---------- Build JSON ----------
temps_flat = []
for j in range(GRID_SIZE):
    for i in range(GRID_SIZE):
        if inside[j, i] and not np.isnan(interp[j, i]):
            temps_flat.append(round(float(interp[j, i]), 1))
        else:
            temps_flat.append(None)

output = {
    "generated": datetime.now(timezone.utc).isoformat(),
    "cols": GRID_SIZE,
    "rows": GRID_SIZE,
    "minTemp": MIN_TEMP,
    "maxTemp": MAX_TEMP,
    "temps": temps_flat,
    "labels": labels,
    "sources": {
        "microclimate": sources.count("mc"),
        "openmeteo":    sources.count("om"),
        "default":      sources.count("default"),
    },
}
with open(OUTPUT_JSON, 'w') as f:
    json.dump(output, f, separators=(',', ':'))

valid = [t for t in temps_flat if t is not None]
print(f"\n✅ Wrote {OUTPUT_JSON}")
print(f"   {len(valid)} cells, {len(labels)} labels")
print(f"   Range: {min(valid):.1f}°F – {max(valid):.1f}°F")
print(f"   Sources: {sources.count('mc')} mc | {sources.count('om')} om | {sources.count('default')} default")