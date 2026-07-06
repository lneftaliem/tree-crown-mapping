"""
prepare_ripley_data.py
Script to prepare tree point clouds and city boundaries for Ripley's K analysis.
Optimized for the Sherlock environment and EXACTLY synchronized with equity analysis filters.

Exported data:
1. [city]_ripley_points.csv: X, Y (meters, UTM), and crown_area.
2. [city]_ripley_boundary.geojson: Study area polygon (meters, UTM).
"""

import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.warp
from skimage.measure import label, regionprops
from scipy.spatial import KDTree
from shapely.geometry import Point, box
from shapely.ops import unary_union
from shapely.strtree import STRtree
from joblib import Parallel, delayed
from tqdm import tqdm

# --- 0. Global Metadata Cache ---
TILE_METADATA_LOOKUP = {}

# --- 1. Sherlock Environment Setup ---
BASE_DIR = os.environ.get("TREE_MAPPING_BASE_DIR", os.getcwd())
if not os.path.exists(BASE_DIR):
    BASE_DIR = os.getcwd() # local fallback

CUTOUTS_DIR = os.path.join(BASE_DIR, "cutouts")
print(f"==========================================")
print(f"ENVIRONMENT DIAGNOSTICS (prepare_ripley_data.py)")
print(f"==========================================")
print(f"Current Working Directory: {os.getcwd()}")
print(f"BASE_DIR: {BASE_DIR} (exists: {os.path.exists(BASE_DIR)})")
print(f"CUTOUTS_DIR: {CUTOUTS_DIR} (exists: {os.path.exists(CUTOUTS_DIR)})")
if os.path.exists(CUTOUTS_DIR):
    print(f"Total files in CUTOUTS_DIR: {len(os.listdir(CUTOUTS_DIR))}")
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_output", "ripley_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Outputs will be saved to: {OUTPUT_DIR}")
print(f"==========================================\n")

# --- 2. Configuration (Synchronized with city_tree_analysis.py) ---
CITIES = ['austin', 'bloomington', 'cupertino', 'surrey']

# Keywords to match tiles in the cutouts directory
CITY_SEARCH_KEYWORDS = {
    'austin':      ['austin'],
    'bloomington': ['bloomington'],
    'cupertino':   ['cupertino'],
    'surrey':      ['surrey'],
}

# Data Cleaning Constants
THRESHOLD = 0.3 
N_WORKERS = 8  

# New Exact Boundaries Configuration
EXACT_BOUNDARIES_DIR = os.path.join(BASE_DIR, 'census_boundaries', 'exact_boundaries')

# Standardized UTM projections (meters) for accurate spatial stats
CITY_UTM_EPSG = {
    'austin':      32614,   # UTM 14N
    'bloomington': 32616,   # UTM 16N
    'cupertino':   32610,   # UTM 10N
    'surrey':      32610,   # UTM 10N
}

def get_city_tiles(city):
    """Filter files in cutouts directory for a specific city, prioritizing pred_ndvi."""
    if not os.path.exists(CUTOUTS_DIR):
        print(f"  [ERROR] Cutouts directory does not exist: {CUTOUTS_DIR}")
        return []
    all_files = os.listdir(CUTOUTS_DIR)
    keywords = [kw.lower() for kw in CITY_SEARCH_KEYWORDS.get(city.lower(), [city.lower()])]
    
    city_raw = []
    for f in all_files:
        if f.startswith('pred_') and f.endswith('.tif') and "_confidence" not in f:
            if any(kw in f.lower() for kw in keywords):
                city_raw.append(f)
    
    # Prioritize 'pred_ndvi_' over 'pred_pan_'
    base_best = {}
    for f in city_raw:
        base = f.replace('pred_pan_', '').replace('pred_ndvi_', '').replace('.tif', '')
        if base not in base_best or f.startswith('pred_ndvi_'):
            base_best[base] = f
                
    res = [os.path.join(CUTOUTS_DIR, f) for f in base_best.values()]
    
    # Detailed diagnostics
    print(f"  [Diagnostics for {city.upper()} in get_city_tiles]:")
    print(f"    Keywords searched: {keywords}")
    print(f"    Total files in cutouts directory: {len(all_files)}")
    print(f"    Matching files (pred_*, non-confidence): {len(city_raw)}")
    ndvi_c = sum(1 for f in city_raw if f.startswith('pred_ndvi_'))
    pan_c = sum(1 for f in city_raw if f.startswith('pred_pan_'))
    print(f"    - NDVI count: {ndvi_c}")
    print(f"    - PAN count: {pan_c}")
    print(f"    Unique base tiles (deduplicated): {len(res)}")
    if res:
        print(f"    - Sample matched files (first 3): {[os.path.basename(p) for p in res[:3]]}")
        
    return res

def process_single_tile(tile_path):
    """Detect tree centroids and convert to Geographic (Lon/Lat) coordinates with robust georeferencing."""
    try:
        filename = os.path.basename(tile_path)
        with rasterio.open(tile_path) as src:
            img = src.read(1).astype(np.float32)
            if img.max() > 1: img /= 255.0
            binary = img > THRESHOLD
            labeled = label(binary.astype(np.uint8), connectivity=1)
            props = regionprops(labeled)
            
            # Resolve georeferencing
            # 1. Check file CRS
            tile_crs = src.crs
            tile_transform = src.transform
            tile_bounds = src.bounds
            
            # 2. Fallback to metadata if file lacks georeferencing (pixel bounds Detected)
            if tile_crs is None or (tile_bounds[0] == 0 and tile_bounds[2] == src.width):
                global TILE_METADATA_LOOKUP
                if not TILE_METADATA_LOOKUP:
                    curr_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
                    for possible_dir in [BASE_DIR, curr_dir, os.getcwd()]:
                        meta_path = os.path.join(possible_dir, 'tile_metadata.csv')
                        if os.path.exists(meta_path):
                            try:
                                df_meta = pd.read_csv(meta_path)
                                TILE_METADATA_LOOKUP = df_meta.set_index('filename').to_dict('index')
                                break
                            except Exception:
                                pass
                m = TILE_METADATA_LOOKUP.get(filename)
                if m:
                    # Construct box from metadata
                    l, b, r, t = m['left'], m['bottom'], m['right'], m['top']
                    tile_bounds = box(l, b, r, t).bounds # use proper tuple
                    tile_crs = 'EPSG:4326'
                    # Construct transform manually
                    tile_transform = rasterio.transform.from_bounds(l, b, r, t, src.width, src.height)
                else:
                    # Final fallback if all else fails
                    tile_crs = 'EPSG:4326'
            
            # 3. Rectify bounds for spatial filter (ensure EPSG:4326)
            b = tile_bounds # (left, bottom, right, top)
            if str(tile_crs).upper() != 'EPSG:4326':
                try:
                    # Reproject bounds to 4326
                    l, b_val, r, t = rasterio.warp.transform_bounds(tile_crs, 'EPSG:4326', b[0], b[1], b[2], b[3])
                    tile_geom = box(l, b_val, r, t)
                except Exception:
                    tile_geom = box(b[0], b[1], b[2], b[3])
            else:
                tile_geom = box(b[0], b[1], b[2], b[3])

            # 4. Extract points and reproject if necessary
            results = []
            xs_raw = []
            ys_raw = []
            areas = []
            for p in props:
                row, col = p.centroid
                x, y = rasterio.transform.xy(tile_transform, row, col)
                xs_raw.append(x)
                ys_raw.append(y)
                areas.append(p.area)
            
            if xs_raw:
                if str(tile_crs).upper() != 'EPSG:4326':
                    # Reproject array of points to 4326
                    xs_4326, ys_4326 = rasterio.warp.transform(tile_crs, 'EPSG:4326', xs_raw, ys_raw)
                    for x, y, a in zip(xs_4326, ys_4326, areas):
                        results.append((x, y, a, filename))
                else:
                    for x, y, a in zip(xs_raw, ys_raw, areas):
                        results.append((x, y, a, filename))
            
            return results, tile_geom
            
    except Exception as e:
        print(f"Error processing {os.path.basename(tile_path)}: {e}")
        return [], None

def prepare_city_data(city):
    print(f"\n>>> Preparing EXACT MATCH Ripley Data for: {city.upper()} <<<")
    
    # Load metadata at start of city processing
    global TILE_METADATA_LOOKUP
    curr_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    meta_path = None
    for possible_dir in [BASE_DIR, curr_dir, os.getcwd()]:
        p = os.path.join(possible_dir, 'tile_metadata.csv')
        if os.path.exists(p):
            meta_path = p
            break
            
    if meta_path:
        try:
            df_meta = pd.read_csv(meta_path)
            # filenames are relative to cutouts dir
            TILE_METADATA_LOOKUP = df_meta.set_index('filename').to_dict('index')
            print(f"  Loaded metadata for {len(TILE_METADATA_LOOKUP)} tiles from {meta_path}.")
        except Exception as e:
            print(f"  Warning: Could not load tile_metadata.csv: {e}")
    
    # 1. Load EXACT Study Boundaries
    # -----------------------------------------------------------
    place_path = os.path.join(EXACT_BOUNDARIES_DIR, f"{city}_exact_boundary.geojson")
    gdf_place = gpd.read_file(place_path)
    if gdf_place.crs != 'EPSG:4326': gdf_place = gdf_place.to_crs('EPSG:4326')
    place_poly = unary_union(gdf_place.geometry)

    census_path = os.path.join(EXACT_BOUNDARIES_DIR, f"{city}_exact_block_groups.geojson")
    gdf_census = gpd.read_file(census_path)
    if gdf_census.crs != 'EPSG:4326': gdf_census = gdf_census.to_crs('EPSG:4326')
    
    # Initial selection: census units intersecting the place bounding box (ensures enclaves are caught)
    from shapely.validation import make_valid
    place_poly = make_valid(place_poly)
    gdf_census['geometry'] = gdf_census.geometry.map(lambda g: make_valid(g) if not g.is_valid else g)
    
    gdf_census = gdf_census[gdf_census.geometry.intersects(box(*place_poly.bounds))].reset_index(drop=True)
    geoid_col = next((c for c in ('GEOID', 'census_id', 'DAUID', 'GEO_ID') if c in gdf_census.columns), 'index')
    if geoid_col == 'index': gdf_census['index'] = gdf_census.index.astype(str)

    # 2. Extract and Deduplicate Points
    # --------------------------------
    tile_paths = get_city_tiles(city)
    # Parallel processing with joblib (loky backend is GIL-free for raster processing)
    results_par = Parallel(n_jobs=N_WORKERS, backend='loky')(
        delayed(process_single_tile)(tp) for tp in tqdm(tile_paths, desc="Detecting Trees")
    )
    
    all_raw_points = []
    tile_geoms = []
    for res, t_geom in results_par:
        all_raw_points.extend(res)
        if t_geom: tile_geoms.append(t_geom)

    if not all_raw_points:
        print("  No trees detected.")
        return

    # Spatial Deduplication (Restricted to Overlaps)
    # ---------------------------------------------
    pts_arr = np.array([(p[0], p[1]) for p in all_raw_points])
    areas_arr = np.array([p[2] for p in all_raw_points])
    tile_ids = [p[3] for p in all_raw_points]
    
    epsilon = 0.00001  # 1.0 meters
    tree_idx = KDTree(pts_arr)
    pairs = tree_idx.query_pairs(epsilon)
    
    to_remove = set()
    for idx1, idx2 in pairs:
        # Only deduplicate if points come from DIFFERENT tiles (effectively restring to overlaps)
        if tile_ids[idx1] != tile_ids[idx2]:
            if idx1 not in to_remove: 
                to_remove.add(idx2)
    
    mask = np.ones(len(pts_arr), dtype=bool)
    mask[list(to_remove)] = False
    
    unique_points_gdf = gpd.GeoDataFrame({
        'geometry': [Point(p[0], p[1]) for p in pts_arr[mask]],
        'crown_area_px': areas_arr[mask]
    }, crs='EPSG:4326')

    # 3. Apply EXACT MATCH Cleaning Filters
    # ------------------------------------
    print(f"  Calculating per-unit statistics for {len(gdf_census)} neighborhoods...")
    
    # A. Count trees per census unit (ensuring 1:1 assignment for trees on boundaries)
    sj = gpd.sjoin(unique_points_gdf, gdf_census[[geoid_col, 'geometry']], how='inner', predicate='intersects')
    # If a point touches multiple units, assign to the first one only
    sj = sj[~sj.index.duplicated(keep='first')]
    counts = sj.groupby(geoid_col).size().to_dict()
    
    # B. Calculate area from tiles (Full area for each tile intersecting the neighborhood)
    # Optimized using STRtree for spatial index (replaces O(N*M) loop)
    tile_size_meters = 256
    tile_area_km2 = (tile_size_meters / 1000) ** 2 # ~0.0655 km2
    
    print(f"  Building spatial index for {len(tile_geoms)} tiles...")
    tile_tree = STRtree(tile_geoms)
    unit_areas = {}
    
    for row in gdf_census.itertuples():
        gid = getattr(row, geoid_col)
        geom = make_valid(row.geometry) if not row.geometry.is_valid else row.geometry
        potential_indices = tile_tree.query(geom)
        
        # Verify actual intersection
        n_intersect = 0
        if potential_indices.size > 0:
            for idx in potential_indices:
                if geom.intersects(tile_geoms[idx]):
                    n_intersect += 1
                    
        unit_areas[gid] = n_intersect * tile_area_km2

    # C. Apply Filters
    kept_geoids = []
    for row in gdf_census.itertuples():
        gid = getattr(row, geoid_col)
        n = counts.get(gid, 0)
        a = unit_areas.get(gid, 0)
        density = n / a if a > 0 else 0
        
        # Include all units intersecting the city footprint
        kept_geoids.append(gid)
    
    print(f"  Filtering: {len(gdf_census)} → {len(kept_geoids)} units pass quality checks.")
    clean_gdf = gdf_census[gdf_census[geoid_col].isin(kept_geoids)]

    # 4. Final Clipping and Reprojection
    # ---------------------------------
    # Optimized point-in-poly filter using sjoin
    print(f"  Clipping {len(unique_points_gdf)} trees to final study area...")
    obs_window_gdf = gpd.GeoDataFrame({'geometry': [place_poly]}, crs='EPSG:4326')
    points_final_gdf = gpd.sjoin(unique_points_gdf, obs_window_gdf, how='inner', predicate='within')
    points_final = points_final_gdf[['geometry', 'crown_area_px']]
    print(f"  Final synchronized tree count: {len(points_final)}")

    utm_crs = f"EPSG:{CITY_UTM_EPSG[city]}"
    points_utm = points_final.to_crs(utm_crs)
    window_utm_gdf = gpd.GeoDataFrame({'geometry': [place_poly]}, crs='EPSG:4326').to_crs(utm_crs)

    # 5. Export
    points_data = pd.DataFrame({
        'x_meters': points_utm.geometry.x,
        'y_meters': points_utm.geometry.y,
        'crown_area_px': points_utm['crown_area_px']
    })
    csv_path = os.path.join(OUTPUT_DIR, f"{city}_ripley_points.csv")
    points_data.to_csv(csv_path, index=False)
    
    geojson_path = os.path.join(OUTPUT_DIR, f"{city}_ripley_boundary.geojson")
    window_utm_gdf.to_file(geojson_path, driver='GeoJSON')
    
    print(f"  SUCCESS: Data exported for {city}")

if __name__ == "__main__":
    for city in CITIES:
        prepare_city_data(city)
