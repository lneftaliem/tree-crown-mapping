import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import KDTree
from joblib import Parallel, delayed
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import functions from prepare_ripley_data without executing __main__
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)
sys.path.append(os.getcwd())
import prepare_ripley_data as prd

def process_tile_wrapper(tile_path, metadata_lookup):
    import prepare_ripley_data as prd
    if not prd.TILE_METADATA_LOOKUP and metadata_lookup:
        prd.TILE_METADATA_LOOKUP = metadata_lookup
    return prd.process_single_tile(tile_path)

THRESHOLDS_M = [0.8, 1.0, 1.5]
CITIES = prd.CITIES

# Helper to calculate average latitude
def get_city_lat(city):
    place_path = os.path.join(prd.EXACT_BOUNDARIES_DIR, f"{city}_exact_boundary.geojson")
    if os.path.exists(place_path):
        gdf = gpd.read_file(place_path)
        utm_epsg = prd.CITY_UTM_EPSG.get(city.lower())
        if utm_epsg:
            # Re-project to projected local UTM CRS first to get an accurate centroid without warnings
            gdf_utm = gdf.to_crs(f"EPSG:{utm_epsg}")
            centroids_4326 = gdf_utm.geometry.centroid.to_crs('EPSG:4326')
            return centroids_4326.y.mean()
        else:
            # Fallback to geographic CRS but suppress the UserWarning
            if gdf.crs != 'EPSG:4326':
                gdf = gdf.to_crs('EPSG:4326')
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                return gdf.geometry.centroid.y.mean()
    else:
        # Fallback approximate latitudes
        fallbacks = {'austin': 30.26, 'bloomington': 39.16, 'cupertino': 37.32, 'surrey': 49.19}
        return fallbacks.get(city.lower(), 40.0)

results_data = []

for city in CITIES:
    print(f"\n--- Sensitivity Analysis for {city.upper()} ---")
    
    # 1. Load exact study boundaries to define the observation window
    place_path = os.path.join(prd.EXACT_BOUNDARIES_DIR, f"{city}_exact_boundary.geojson")
    if not os.path.exists(place_path):
        print(f"Boundary missing for {city}, skipping.")
        continue
    gdf_place = gpd.read_file(place_path)
    if gdf_place.crs != 'EPSG:4326': 
        gdf_place = gdf_place.to_crs('EPSG:4326')
    from shapely.ops import unary_union
    from shapely.validation import make_valid
    place_poly = unary_union(gdf_place.geometry)
    place_poly = make_valid(place_poly)
    obs_window_gdf = gpd.GeoDataFrame({'geometry': [place_poly]}, crs='EPSG:4326')
    
    lat = get_city_lat(city)
    print(f"  City centroid latitude: {lat:.3f}°")
    
    # 2. Extract raw points
    tile_paths = prd.get_city_tiles(city)
    
    # Populate metadata lookup
    meta_path = None
    for possible_dir in [prd.BASE_DIR, script_dir, os.getcwd()]:
        p = os.path.join(possible_dir, 'tile_metadata.csv')
        if os.path.exists(p):
            meta_path = p
            break
            
    if meta_path and not prd.TILE_METADATA_LOOKUP:
        try:
            df_meta = pd.read_csv(meta_path)
            prd.TILE_METADATA_LOOKUP = df_meta.set_index('filename').to_dict('index')
        except Exception as e:
            print(f"  Warning: Could not load tile_metadata.csv from {meta_path}: {e}")
    
    print(f"  Detecting trees across {len(tile_paths)} tiles...")
    results_par = Parallel(n_jobs=prd.N_WORKERS, backend='loky')(
        delayed(process_tile_wrapper)(tp, prd.TILE_METADATA_LOOKUP) for tp in tqdm(tile_paths, desc="Detecting Trees")
    )
    
    all_raw_points = []
    for res, t_geom in results_par:
        all_raw_points.extend(res)
        
    if not all_raw_points:
        print(f"  No trees detected for {city}.")
        continue
        
    pts_arr = np.array([(p[0], p[1]) for p in all_raw_points])
    areas_arr = np.array([p[2] for p in all_raw_points])
    tile_ids = [p[3] for p in all_raw_points]
    
    print(f"  Total raw points detected: {len(pts_arr)}")
    
    # Build KDTree once per city
    tree_idx = KDTree(pts_arr)
    
    # To match prepare_ripley_data.py EXACTLY:
    # In prepare_ripley_data.py, the deduplication threshold of 1.0m is mapped to a hardcoded 0.00001 degrees.
    # Therefore, to replicate this exact tree-counting standard, we scale epsilon directly in degrees 
    # as 0.00001 degrees per meter (so 0.8m -> 8e-6, 1.0m -> 1e-5, 1.5m -> 1.5e-5).
    city_results = []
    
    for thresh_m in THRESHOLDS_M:
        epsilon = thresh_m * 0.00001
        
        # Query KDTree
        pairs = tree_idx.query_pairs(epsilon)
        
        to_remove = set()
        for idx1, idx2 in pairs:
            if tile_ids[idx1] != tile_ids[idx2]:
                if idx1 not in to_remove:
                    to_remove.add(idx2)
                    
        mask = np.ones(len(pts_arr), dtype=bool)
        mask[list(to_remove)] = False
        
        n_unique_pre_clip = np.sum(mask)
        
        # Create unique points GDF
        from shapely.geometry import Point
        unique_points_gdf = gpd.GeoDataFrame({
            'geometry': [Point(p[0], p[1]) for p in pts_arr[mask]]
        }, crs='EPSG:4326')
        
        # Spatial clip to the place boundary
        points_final_gdf = gpd.sjoin(unique_points_gdf, obs_window_gdf, how='inner', predicate='within')
        n_final = len(points_final_gdf)
        
        print(f"    Threshold {thresh_m}m -> epsilon {epsilon:.7f} deg -> {n_final} trees retained")
        
        city_results.append({
            'city': city.title(),
            'threshold_m': thresh_m,
            'n_trees': n_final
        })
        
    results_data.extend(city_results)

# --- 3. Plotting ---
print("\nGenerating 2x2 grid plot...")
df_res = pd.DataFrame(results_data)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

# Colors matching the rest of the paper's figures
OI_COLORS = {
    'austin':      '#E69F00',
    'bloomington': '#56B4E9',
    'cupertino':   '#009E73',
    'surrey':      '#CC79A7'
}

CITY_LABELS = {
    'austin':      'Austin, TX',
    'bloomington': 'Bloomington, IN',
    'cupertino':   'Cupertino, CA',
    'surrey':      'Surrey, BC'
}

# Pre-calculate global min/max percent retained across all cities (relative to the 1.0m standard baseline)
all_pcounts = []
for city in CITIES:
    if not df_res.empty and 'city' in df_res.columns:
        city_df = df_res[df_res['city'] == city.title()]
    else:
        city_df = pd.DataFrame()
    if len(city_df) > 0:
        count_10 = city_df[city_df['threshold_m'] == 1.0]['n_trees'].values[0]
        pcts = city_df['n_trees'].values / count_10 * 100
        all_pcounts.extend(pcts)

if all_pcounts:
    global_min_pct = min(all_pcounts)
    global_max_pct = max(all_pcounts)
    # Small padding of 0.2% around the min/max to show detailed variation
    y_min = global_min_pct - 0.2
    y_max = global_max_pct + 0.2
else:
    y_min, y_max = 99.0, 101.0

import matplotlib.ticker as ticker

for i, city in enumerate(CITIES):
    ax = axes[i]
    if not df_res.empty and 'city' in df_res.columns:
        city_df = df_res[df_res['city'] == city.title()].copy()
    else:
        city_df = pd.DataFrame()
        
    if len(city_df) > 0:
        count_10 = city_df[city_df['threshold_m'] == 1.0]['n_trees'].values[0]
        city_df['pct_retained'] = city_df['n_trees'] / count_10 * 100
        
        color = OI_COLORS.get(city.lower(), 'b')
        ax.plot(city_df['threshold_m'], city_df['pct_retained'], marker='o', linestyle='-', color=color, linewidth=2, markersize=8)
        ax.set_title(CITY_LABELS.get(city.lower(), city.title()), fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('Deduplication Threshold (meters)', fontsize=10)
        ax.set_ylabel('Unique Trees Retained (%)', fontsize=10)
        ax.set_xticks(THRESHOLDS_M)
        ax.set_ylim(y_min, y_max)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=2))
        ax.ticklabel_format(useOffset=False, style='plain', axis='x')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add a text annotation in each panel showing the baseline N at 1.0m (the standard)
        if count_10 > 0:
            from math import log10, floor
            rounded_n = int(round(count_10, -floor(log10(count_10)) + 2))
        else:
            rounded_n = 0
        label_text = f"Standard N = {rounded_n:,} at 1.0m"
        ax.text(0.95, 0.95, label_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                color='dimgray', fontsize=9, fontweight='semibold')
    else:
        ax.set_title(city.title() + ' (No Data)')
        ax.axis('off')

plt.tight_layout()
out_path = os.path.join(os.getcwd(), 'sensitivity_analysis_dedup.png')
plt.savefig(out_path, dpi=300)
print(f"Saved plot to {out_path}")
