"""
tree_size_vs_income.py
Analyzes the relationship between median tree crown area and median household income per census unit.
Provides empirical evidence for the 'Structural Maturity Gap' in urban forests.

UNIT OF ANALYSIS:
(a) Correlation is between median household income and median individual crown area per census unit.
(b) Each data point in the correlation represents one census unit (e.g., Block Group or Dissemination Area).
"""

import os
import sys
import gc
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from affine import Affine
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from scipy.spatial import KDTree
from shapely.geometry import Point, box
from shapely.ops import unary_union
from math import cos, radians
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from skimage.measure import label, regionprops
import matplotlib
matplotlib.use('Agg')
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from scipy import stats
from scipy.stats import spearmanr

# ============================================================================
# CONFIGURATION
# ============================================================================
# Detect Environment
SHERLOCK_BASE = os.environ.get("TREE_MAPPING_BASE_DIR", os.getcwd())
DROPBOX_BASE = os.environ.get("TREE_MAPPING_BASE_DIR", os.getcwd())

if os.path.exists(SHERLOCK_BASE):
    BASE_DIR = SHERLOCK_BASE
    print(f"Environment: Sherlock detected. Using BASE_DIR: {BASE_DIR}")
    # On Sherlock, these are directly in BASE_DIR or match the script's established structure
    CUTOUTS_DIR = os.path.join(BASE_DIR, "cutouts")
    CENSUS_SUBDIR = "census_boundaries"
    INCOME_SUBDIR = "income_data"
else:
    BASE_DIR = DROPBOX_BASE
    print(f"Environment: Local/Dropbox detected. Using BASE_DIR: {BASE_DIR}")
    # Locally, data is organized in a 'notebooks' folder
    CUTOUTS_DIR = os.path.join(BASE_DIR, "notebooks", "cutouts")
    CENSUS_SUBDIR = os.path.join("notebooks", "census_boundaries")
    INCOME_SUBDIR = os.path.join("notebooks", "income_data")

OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# LST Data Path (relative to BASE_DIR)
LST_DATA_DIR = os.path.join(BASE_DIR, "temp_data") if os.path.exists(SHERLOCK_BASE) else os.path.join(BASE_DIR, "notebooks", "temp_data")
RAW_IMAGES_DIR = os.path.join(BASE_DIR, "raw_images") if os.path.exists(SHERLOCK_BASE) else os.path.join(BASE_DIR, "notebooks", "raw_images")

CENSUS_BOUNDARY_PATHS = {
    'austin': os.path.join(BASE_DIR, CENSUS_SUBDIR, 'tl_2023_48_bg', 'tl_2023_48_bg.shp'),
    'bloomington': os.path.join(BASE_DIR, CENSUS_SUBDIR, 'tl_2023_18_bg', 'tl_2023_18_bg.shp'),
    'cupertino': os.path.join(BASE_DIR, CENSUS_SUBDIR, 'tl_2023_06_bg', 'tl_2023_06_bg.shp'),
    'surrey': os.path.join(BASE_DIR, CENSUS_SUBDIR, 'lda_000b21a_e', 'lda_000b21a_e.shp'),
}

US_INCOME_DATA_PATH = os.path.join(BASE_DIR, INCOME_SUBDIR, 'ACSDT5Y2023', 'ACSDT5Y2023.B19013-Data.csv')
CA_INCOME_DATA_PATH = os.path.join(BASE_DIR, INCOME_SUBDIR, '98-401-X2021006_BC_CB_eng_CSV', '98-401-X2021006_English_CSV_data_BritishColumbia.csv')

CAD_TO_USD = 1.37  # Conversion rate for Surrey

# County/Region filters (matching city_tree_analysis.py)
CITY_FILTERS = {
    'austin': {'col': 'COUNTYFP', 'val': '453'},       # Travis County
    'bloomington': {'col': 'COUNTYFP', 'val': '105'},  # Monroe County
    'cupertino': {'col': 'COUNTYFP', 'val': '085'},    # Santa Clara County
    'surrey': {'col': 'CSDUID', 'val': '5915004'},     # Surrey CSD
}

# Matplotlib defaults for Nature Cities
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42
})

# ============================================================================
# UTILITIES
def load_tile_metadata(meta_path):
    """Load tile bounds from CSV metadata."""
    if not os.path.exists(meta_path): return {}
    try:
        df = pd.read_csv(meta_path)
        lookup = {}
        for r in df.itertuples(index=False):
            fn = getattr(r, 'filename', None)
            if fn:
                lookup[fn] = box(getattr(r, 'left'), getattr(r, 'bottom'), getattr(r, 'right'), getattr(r, 'top'))
        return lookup
    except: return {}

def get_base_name(filename):
    """Extract coordinate base from tile filename."""
    return filename.replace('pred_pan_', '').replace('pred_ndvi_', '').replace('pan_', '').replace('ndvi_', '').replace('_confidence', '').replace('.tif', '')

def find_source_images(raw_dir):
    """Find all source TIF files and extract their georeference info."""
    source_info = {}
    if not os.path.exists(raw_dir):
        return source_info
        
    for root, dirs, files in os.walk(raw_dir):
        for f in files:
            if f.endswith('reprojected.tif'):
                filepath = os.path.join(root, f)
                parts = f.replace('.tif', '').split('_')
                city = parts[1].lower()
                tile_id = None
                for i, part in enumerate(parts):
                    if part.isdigit() and len(part) == 3 and i + 1 < len(parts):
                        if i+1 < len(parts) and parts[i+1].isdigit() and len(parts[i+1]) == 3:
                            tile_id = f"{part}_{parts[i+1]}"
                            break
                if tile_id:
                    try:
                        with rasterio.open(filepath) as src:
                            source_info[(city, tile_id)] = {'transform': src.transform, 'crs': src.crs}
                    except: pass
    return source_info

def calculate_cutout_bounds(source_info, cutout_filename, cutout_width=5000, cutout_height=5000):
    """Calculate geographic bounds and transform for a cutout using source imagery transform."""
    filename_lower = cutout_filename.lower()
    city = next((c for c in ['austin', 'bloomington', 'cupertino', 'surrey'] if c in filename_lower), None)
    if not city: return None
    
    # Parse offsets and tile ID
    parts = cutout_filename.replace('.tif', '').split('_')
    x_offset = y_offset = tile_id = None
    for part in parts:
        if '-' in part:
            offsets = part.split('-')
            if len(offsets) == 2 and offsets[0].isdigit() and offsets[1].isdigit():
                x_offset, y_offset = int(offsets[0]), int(offsets[1])
    for i, part in enumerate(parts):
        if part.isdigit() and len(part) == 3 and i + 1 < len(parts):
            if i+1 < len(parts) and parts[i+1].isdigit() and len(parts[i+1]) == 3:
                tile_id = f"{part}_{parts[i+1]}"
                break
                
    if x_offset is None or tile_id is None: return None
    
    key = (city, tile_id)
    if key in source_info:
        s_trans = source_info[key]['transform']
        left = s_trans.c + x_offset * s_trans.a
        top = s_trans.f + y_offset * s_trans.e
        right = left + cutout_width * s_trans.a
        bottom = top + cutout_height * s_trans.e
        
        # Create affine transform for the cutout
        res_x = s_trans.a
        res_y = s_trans.e
        new_trans = Affine(res_x, s_trans.b, left, s_trans.d, res_y, top)
        
        return box(min(left, right), min(bottom, top), max(left, right), max(bottom, top)), new_trans
    return None

def _get_tile_bounds(path):
    """Get bounds and transform for a georeferenced TIF."""
    try:
        with rasterio.open(path) as src:
            if src.crs is not None:
                b = src.bounds
                geom = box(b.left, b.bottom, b.right, b.top)
                return path, geom, src.transform
    except: pass
    return None
# ============================================================================

def load_us_income_data(filepath):
    """Load and clean US ACS median household income data."""
    if not os.path.exists(filepath):
        return pd.DataFrame()
    df = pd.read_csv(filepath, skiprows=[1])
    # Extract GEOID from "G110001U012345..." format (it's the last 12 chars usually)
    # But usually GEO_ID column has it.
    df['GEOID'] = df['GEO_ID'].str[-12:]
    # Convert income to numeric, handling "250,000+"
    df['median_income'] = df['B19013_001E'].replace(['250,000+', '-', '2,500-'], ['250000', np.nan, '2500'])
    df['median_income'] = pd.to_numeric(df['median_income'], errors='coerce')
    return df[['GEOID', 'median_income']].dropna()

def load_canadian_income_data(filepath, region_prefix='59'):
    """Load CA median household income data for Dissemination Areas."""
    if not os.path.exists(filepath):
        return pd.DataFrame()
    
    # Use chunked reading for large census files
    cols_to_use = ['ALT_GEO_CODE', 'GEO_LEVEL', 'CHARACTERISTIC_ID', 'C1_COUNT_TOTAL']
    chunks = []
    try:
        for chunk in pd.read_csv(filepath, usecols=cols_to_use, chunksize=500000, 
                                 dtype={'ALT_GEO_CODE': str, 'CHARACTERISTIC_ID': int},
                                 encoding='latin-1'):
            # Filter for: Dissemination Area + Median household income (ID 243)
            mask = (chunk['GEO_LEVEL'] == 'Dissemination area') & (chunk['CHARACTERISTIC_ID'] == 243)
            if region_prefix:
                mask = mask & chunk['ALT_GEO_CODE'].str.startswith(region_prefix)
            
            filtered = chunk[mask]
            if not filtered.empty:
                chunks.append(filtered)
    except Exception as e:
        print(f"  Error reading Canadian income data: {e}")
        return pd.DataFrame()

    if not chunks:
        return pd.DataFrame()
    
    df = pd.concat(chunks, ignore_index=True)
    df = df.rename(columns={'ALT_GEO_CODE': 'DAUID', 'C1_COUNT_TOTAL': 'median_income'})
    df['median_income'] = pd.to_numeric(df['median_income'], errors='coerce')
    return df[['DAUID', 'median_income']].dropna()

def load_lst_data(city):
    """Load Landsat Surface Temperature data from shapefiles.
    Expected in notebooks/temp_data/LST_{City}_2025.shp
    """
    city_map = {'austin': 'Austin', 'bloomington': 'Bloomington', 'cupertino': 'Cupertino', 'surrey': 'Surrey'}
    city_name = city_map.get(city.lower(), city.capitalize())
    
    # Try multiple path variations
    search_paths = [
        os.path.join(BASE_DIR, 'notebooks', 'temp_data', f"LST_{city_name}_2025.shp"),
        os.path.join(BASE_DIR, 'temp_data', f"LST_{city_name}_2025.shp"),
        os.path.join(os.getcwd(), 'notebooks', 'temp_data', f"LST_{city_name}_2025.shp")
    ]
    
    lst_path = None
    for p in search_paths:
        if os.path.exists(p):
            lst_path = p
            break
            
    if not lst_path:
        print(f"  Warning: LST shapefile not found for {city_name} in {search_paths[0]}")
        return pd.DataFrame(columns=['GEOID', 'mean_lst'])

    try:
        gdf = gpd.read_file(lst_path)
        # Ensure CRS is 4326 for matching
        if gdf.crs and gdf.crs != 'EPSG:4326':
            gdf = gdf.to_crs('EPSG:4326')
        
        # GEOID column varies: GEOID, DAUID, etc.
        id_col = next((c for c in ('GEOID', 'DAUID') if c in gdf.columns), None)
        if not id_col:
            print(f"  Warning: No GEOID/DAUID column in LST data for {city}")
            return pd.DataFrame(columns=['GEOID', 'mean_lst'])
            
        # Temperature column is typically 'mean'
        temp_col = 'mean' if 'mean' in gdf.columns else ('LST' if 'LST' in gdf.columns else None)
        if not temp_col:
            print(f"  Warning: No temperature column (mean/LST) in LST data for {city}")
            return pd.DataFrame(columns=['GEOID', 'mean_lst'])

        df = pd.DataFrame(gdf[[id_col, temp_col]])
        df.columns = ['GEOID', 'mean_lst']
        df['GEOID'] = df['GEOID'].astype(str)
        return df.dropna()
    except Exception as e:
        print(f"  Error loading LST for {city}: {e}")
        return pd.DataFrame(columns=['GEOID', 'mean_lst'])

def _load_place_boundary(city):
    """Load the official place boundary shapefile for a city for clipping.
    REPLICATING city_tree_analysis.py robustness.
    """
    boundary_configs = {
        'austin':      {'path': 'census_boundaries/tl_2023_48_place/tl_2023_48_place.shp', 'col': 'NAME',   'val': 'Austin'},
        'bloomington': {'path': 'census_boundaries/tl_2023_18_place/tl_2023_18_place.shp', 'col': 'NAME',   'val': 'Bloomington'},
        'cupertino':   {'path': 'census_boundaries/tl_2023_06_place/tl_2023_06_place.shp', 'col': 'NAME',   'val': 'Cupertino'},
        'surrey':      {'path': 'census_boundaries/lcsd000b21a_e/lcsd000b21a_e.shp',      'col': 'CSDNAME', 'val': 'Surrey'}
    }
    cfg = boundary_configs.get(city)
    if not cfg:
        return None
    try:
        # Resolve path
        rel_path = f"notebooks/{cfg['path']}"
        path = os.path.join(os.getcwd(), rel_path)
        if not os.path.exists(path):
            path = os.path.join(BASE_DIR, cfg['path'])
            if not os.path.exists(path):
                path = os.path.join(
                    os.environ.get("TREE_MAPPING_BASE_DIR", os.getcwd()), "notebooks", cfg['path']
                )
        
        if os.path.exists(path):
            gdf_b = gpd.read_file(path)
            if gdf_b.crs and gdf_b.crs != 'EPSG:4326': 
                gdf_b = gdf_b.to_crs('EPSG:4326')
            
            name_filter = cfg['val']
            name_col = cfg['col']
            
            # Use exact match or variations if strictNAME fails
            # Most common Census formats: "Austin city", "Cupertino city"
            variations = [name_filter, f"{name_filter} city", f"City of {name_filter}", f"{name_filter} town"]
            subset = gpd.GeoDataFrame()
            for v in variations:
                m = gdf_b[gdf_b[name_col].str.strip().str.lower() == v.lower()]
                if not m.empty:
                    subset = m
                    break
            
            if subset.empty:
                # Fallback to contains
                subset = gdf_b[gdf_b[name_col].str.contains(name_filter, case=False, na=False)]
            
            if not subset.empty:
                # If multiple (e.g. Austin city vs Austin County), pick largest or "city"
                if len(subset) > 1 and 'NAMELSAD' in subset.columns:
                    city_m = subset[subset['NAMELSAD'].str.contains('city', case=False, na=False)]
                    if not city_m.empty: subset = city_m
                
                geom = unary_union(subset.geometry)
                # Ensure valid
                if not geom.is_valid: geom = geom.buffer(0)
                return geom
    except Exception as e:
        print(f"  Warning: Could not load boundary for {city}: {e}")
    return None

def _get_tile_bounds(tile_path):
    """Get bounds of a tile in EPSG:4326. Robust to missing CRS."""
    try:
        with rasterio.open(tile_path) as src:
            b = src.bounds
            # If coordinates are like -180 to 180 and -90 to 90, assume 4326
            if -180.1 <= b.left <= 180.1 and -90.1 <= b.bottom <= 90.1:
                return (tile_path, box(b.left, b.bottom, b.right, b.top))
            return None
    except Exception:
        return None

def run_mediation(df, city_name, n_boot=5000):
    """Run formal mediation analysis with bootstrapping."""
    import statsmodels.api as sm
    
    # Ensure numeric types (avoid object/bool dtype issues with statsmodels)
    X = pd.to_numeric(df['income_std'], errors='coerce')
    M = pd.to_numeric(df['tree_density_log_std'], errors='coerce')
    Y = pd.to_numeric(df['lst_std'], errors='coerce')
    
    # Path a: X -> M
    model_a = sm.OLS(M, sm.add_constant(X)).fit()
    a = model_a.params['income_std']
    p_a = model_a.pvalues['income_std']
    
    # Path b and c': M -> Y and X -> Y (controlling for each other)
    model_bc = sm.OLS(Y, sm.add_constant(df[['income_std', 'tree_density_log_std']])).fit()
    b = model_bc.params['tree_density_log_std']
    c_prime = model_bc.params['income_std']
    p_b = model_bc.pvalues['tree_density_log_std']
    p_c_prime = model_bc.pvalues['income_std']
    
    # Total Effect: X -> Y
    model_total = sm.OLS(Y, sm.add_constant(X)).fit()
    total_effect = model_total.params['income_std']
    p_total = model_total.pvalues['income_std']
    
    # Indirect Effect
    indirect_effect = a * b
    
    # Bootstrapping for Indirect Effect CI
    boot_indirect = []
    for _ in range(n_boot):
        boot_df = df.sample(n=len(df), replace=True)
        # We don't re-standardize inside bootstrap to maintain the same scale meaning
        # but the relationships should be estimated on the resampled data
        ma = sm.OLS(boot_df['tree_density_log_std'], sm.add_constant(boot_df['income_std'])).fit()
        mbc = sm.OLS(boot_df['lst_std'], sm.add_constant(boot_df[['income_std', 'tree_density_log_std']])).fit()
        boot_indirect.append(ma.params['income_std'] * mbc.params['tree_density_log_std'])
    
    ci_lower = np.percentile(boot_indirect, 2.5)
    ci_upper = np.percentile(boot_indirect, 97.5)
    
    sig_indirect = ci_lower * ci_upper > 0
    prop_mediated = indirect_effect / total_effect if total_effect != 0 else np.nan
    
    return {
        'City': city_name,
        'a': a, 'p_a': p_a,
        'b': b, 'p_b': p_b,
        'c_prime': c_prime, 'p_c_prime': p_c_prime,
        'Total_Effect': total_effect,
        'Direct_Effect': c_prime,
        'Indirect_Effect': indirect_effect,
        'Indirect_CI_Lower': ci_lower,
        'Indirect_CI_Upper': ci_upper,
        'Proportion_Mediated': prop_mediated,
        'p_total': p_total,
        'p_direct': p_c_prime,
        'significant': sig_indirect
    }

def _get_tile_bounds(tile_path):
    """Get bounds of a tile in EPSG:4326."""
    try:
        with rasterio.open(tile_path) as src:
            if src.crs is None: return None
            # Bounds are already in the CRS of the file (usually EPSG:4326 for these tiles)
            b = src.bounds
            return (tile_path, box(b.left, b.bottom, b.right, b.top))
    except Exception:
        return None

def _analyze_tile_size(args):
    """Extract tree centroids and areas from a single tile using provided transform."""
    tile_path, recovered_transform, threshold = args
    try:
        with rasterio.open(tile_path) as src:
            pred = src.read(1).astype(np.float32)
            if pred.max() > 1: pred /= 255.0
            binary = pred > threshold
            labeled = label(binary, connectivity=1)
            props = regionprops(labeled)
            
            # Use provided transform if available, otherwise fallback to raster's own
            trans = recovered_transform if recovered_transform else src.transform
            
            tree_data = [] # List of (lon, lat, area_m2)
            for p in props:
                row, col = p.centroid
                # Transform pixel coordinates to geographic coordinates
                lon, lat = trans * (col, row)
                area_m2 = p.area * 1.0 
                tree_data.append((lon, lat, area_m2))
            return tree_data
    except Exception:
        return []

# ============================================================================
# MAIN ANALYSIS LOGIC
# ============================================================================

def run_tree_size_analysis(city, cutouts_dir):
    """Run tree size vs income analysis for a city."""
    print(f"\nAnalyzing {city.upper()}...")
    
    # 1. Load Census Boundaries
    shp_path = CENSUS_BOUNDARY_PATHS.get(city)
    if not shp_path or not os.path.exists(shp_path):
        print(f"  Shapefile not found: {shp_path}")
        return None
    
    gdf = gpd.read_file(shp_path)
    
    print(f"  Census CRS: {gdf.crs}, Units: {len(gdf)}")
    if gdf.crs is None or (hasattr(gdf.crs, 'to_epsg') and gdf.crs.to_epsg() != 4326):
        print(f"  Reprojecting census units to EPSG:4326...")
        gdf = gdf.to_crs('EPSG:4326')
    
    # APPLY CITY FILTERS to match city_tree_analysis.py
    if city in CITY_FILTERS:
        f = CITY_FILTERS[city]
        col = f.get('col')
        if col and col in gdf.columns:
            # Ensure string comparison (TIGER data often has leading zeros)
            if 'val' in f:
                v = str(f['val'])
                gdf = gdf[gdf[col].astype(str) == v]
            elif 'prefix' in f:
                p = str(f['prefix'])
                gdf = gdf[gdf[col].astype(str).str.startswith(p)]
        else:
            print(f"  Warning: CITY_FILTERS column '{col}' not in census_gdf — keeping all {len(gdf)} units")

    # Clip census units to place boundary (matching density map)
    city_boundary = _load_place_boundary(city)
    if city_boundary and not city_boundary.is_empty:
        # Re-verify and Fix/Reproject GDF after filter
        if gdf.crs and gdf.crs != 'EPSG:4326':
            gdf = gdf.to_crs('EPSG:4326')
        
        _before_pb = len(gdf)
        # Use intersects to find block groups that are part of the city
        gdf = gdf[gdf.geometry.intersects(city_boundary)]
        print(f"  Clipped census units to place boundary: {_before_pb} -> {len(gdf)}")
    
    geoid_col = next((c for c in ('GEOID', 'DAUID') if c in gdf.columns), 'GEOID')
    
    # 2. Find Tiles and Catalog Georeferencing
    all_files = os.listdir(cutouts_dir)
    raw_files = [f for f in all_files if f.endswith('.tif') and city in f.lower()]
    
    # [MODIFIED] Catalog georeferencing using metadata, siblings, and source images
    source_info = find_source_images(RAW_IMAGES_DIR)
    tile_meta = load_tile_metadata('tile_metadata.csv')
    geo_catalog = {} 
    
    for f in raw_files:
        base = get_base_name(f)
        path = os.path.join(cutouts_dir, f)
        geom = trans = None
        
        # Try both Affine Recovery (Priority for consistency) and embedded
        res_val = calculate_cutout_bounds(source_info, f)
        if res_val:
            geom, trans = res_val
            
        if geom is None:
            bounds_info = _get_tile_bounds(path)
            if bounds_info:
                _, geom, trans = bounds_info
        
        if geom is None:
            if f in tile_meta:
                geom = tile_meta[f]
            
        if geom:
            if base not in geo_catalog:
                geo_catalog[base] = (geom, trans)
                
    # Now select relevant prediction tiles and inject recovered bounds
    city_tiles_data = []
    for f in raw_files:
        if ('pred_pan' in f or 'pred_ndvi' in f) and '_confidence' not in f:
            base = get_base_name(f)
            path = os.path.join(cutouts_dir, f)
            
            entry = geo_catalog.get(base)
            if entry:
                geom, trans = entry
                city_tiles_data.append((path, geom, trans))

    if not city_tiles_data:
        print(f"  No georeferenced tiles found for {city} (even after source recovery)")
        return None
    
    tile_paths, tile_geoms, tile_transforms = zip(*city_tiles_data)
    tiles_gdf = gpd.GeoDataFrame({'path': tile_paths, 'transform': tile_transforms}, geometry=list(tile_geoms), crs='EPSG:4326')
    
    print(f"  Cataloged {len(tiles_gdf)} tiles with recovered georeferencing.")
    print(f"  Tiles GDF Bounds: {tiles_gdf.total_bounds}")
    print(f"  Census GDF Bounds: {gdf.total_bounds}")
    
    # Fast spatial join to find tiles that intersect ANY census unit
    # Keep track of the transform for each tile in the sjoin result
    intersecting_tiles = gpd.sjoin(tiles_gdf, gdf[['geometry']], how='inner', predicate='intersects')
    
    # Create arguments for pool.map: (path, transform, threshold)
    map_args = []
    seen_paths = set()
    for _, row in intersecting_tiles.iterrows():
        if row['path'] not in seen_paths:
            map_args.append((row['path'], row['transform'], 0.3))
            seen_paths.add(row['path'])
    
    print(f"  {len(map_args)}/{len(tiles_gdf)} tiles intersect census units. Processing...")
    
    if not map_args:
        return None

    # 4. Process Tiles in Parallel
    with ProcessPoolExecutor(max_workers=16) as pool:
        all_tree_data_chunks = list(tqdm(pool.map(_analyze_tile_size, map_args), total=len(map_args), desc="Analyzing Tiles"))
    
    all_tree_data = [item for sublist in all_tree_data_chunks for item in sublist]
    if not all_tree_data:
        print("  No trees detected")
        return None
    
    # 5. Global Deduplication
    pts = np.array(all_tree_data) # [lon, lat, area]
    tree_idx = KDTree(pts[:, :2])
    epsilon = 0.00002 # ~2 meters
    
    # Vectorized pairs query
    pairs = list(tree_idx.query_pairs(epsilon))
    
    to_remove = set()
    for idx1, idx2 in pairs:
        if idx1 not in to_remove:
            to_remove.add(idx2)
    
    # Use boolean indexing for speed
    mask = np.ones(len(pts), dtype=bool)
    if to_remove:
        mask[list(to_remove)] = False
    unique_trees = pts[mask]
    print(f"  Unique trees: {len(unique_trees)}")
    
    # 6. Spatial Join to Census Units
    tree_df = pd.DataFrame(unique_trees, columns=['lon', 'lat', 'area_m2'])
    tree_gdf = gpd.GeoDataFrame(tree_df, geometry=gpd.points_from_xy(tree_df.lon, tree_df.lat), crs='EPSG:4326')
    
    joined = gpd.sjoin(tree_gdf, gdf[[geoid_col, 'geometry']], how='inner', predicate='intersects')
    
    # 7. Aggregate Tree Size per Unit
    # UNIT OF ANALYSIS: Census Unit (GEOID)
    # Correlation is between median household income and median individual crown area per census unit.
    total_crown_area = joined.groupby(geoid_col)['area_m2'].sum().reset_index()
    total_crown_area.columns = ['GEOID', 'total_crown_area_m2']
    
    # SWITCHED TO MEDIAN: Robust to outliers and parkland parcels
    median_crown_area = joined.groupby(geoid_col)['area_m2'].median().reset_index()
    median_crown_area.columns = ['GEOID', 'median_crown_area_m2']
    
    # Ensure geoid_col is handled carefully to avoid duplication
    size_stats = total_crown_area.merge(median_crown_area, on='GEOID')
    print(f"  Mapped trees to {len(size_stats)} units.")
    
    # Calculate area of each unit in hectares
    # Use a meter-based projection for accurate area
    gdf_meter = gdf.to_crs('EPSG:3857') 
    gdf['unit_area_ha'] = gdf_meter.geometry.area / 10000.0
    
    # CRITICAL FIX: Ensure geoid_col is unique in units before merging
    unit_areas = gdf[[geoid_col, 'unit_area_ha']].drop_duplicates(subset=[geoid_col])
    unit_areas.columns = ['GEOID', 'unit_area_ha'] 
    
    # Tree count for density (trees/km2)
    tree_counts = joined.groupby(geoid_col).size().reset_index()
    tree_counts.columns = ['GEOID', 'tree_count']
    
    # Inner join to only keep units with both canopy and area data
    size_stats = pd.merge(size_stats, unit_areas, on='GEOID', how='inner')
    size_stats = pd.merge(size_stats, tree_counts, on='GEOID', how='inner')
    
    size_stats['crown_area_per_ha'] = size_stats['total_crown_area_m2'] / size_stats['unit_area_ha']
    size_stats['tree_density_km2'] = size_stats['tree_count'] / (size_stats['unit_area_ha'] / 100.0)
    
    # Filter out units with improbable canopy cover (>100% or unrealistic)
    # 1 ha = 10,000 m2. Canopy cover should not exceed total unit area significantly
    # unless there's overlapping structure (rare), but we cap at 1000 m2/ha * 10 = 10k? No.
    # m2/ha = (total m2) / (total ha). Max possible is 10,000 m2/ha if 100% covered.
    size_stats = size_stats[size_stats['crown_area_per_ha'] <= 10000.0]
    
    # 7. Load Income Data
    if city == 'surrey':
        income_df = load_canadian_income_data(CA_INCOME_DATA_PATH)
        income_df.columns = ['GEOID', 'median_income']
        # Convert CAD to USD
        income_df['median_income'] /= CAD_TO_USD
    else:
        income_df = load_us_income_data(US_INCOME_DATA_PATH)
    
    income_df = income_df.drop_duplicates(subset=['GEOID']) # Safety check
    
    # 8. Merge
    merged = pd.merge(size_stats, income_df, on='GEOID', how='inner')
    
    # 9. Load and Join LST Data
    lst_df = load_lst_data(city)
    if not lst_df.empty:
        # Diagnostic: check IDs before merge
        n_before = len(merged)
        # Ensure string comparison
        merged['GEOID'] = merged['GEOID'].astype(str)
        lst_df['GEOID'] = lst_df['GEOID'].astype(str)
        
        merged = pd.merge(merged, lst_df, on='GEOID', how='inner')
        n_after = len(merged)
        print(f"  LST Join: {n_after}/{n_before} units retained after joining temperature data.")
        
        if n_after == 0 and n_before > 0:
            print("  DEBUG: GEOID mismatch detected.")
            print(f"    Merged (target) samples: {merged['GEOID'].head(3).tolist() if not merged.empty else size_stats['GEOID'].head(3).tolist()}")
            print(f"    LST CSV (source) samples: {lst_df['GEOID'].head(3).tolist()}")
            
        if n_before > n_after:
            print(f"  Removed {n_before - n_after} units with missing LST values for {city}.")
    else:
        print(f"  Warning: No LST data joined for {city}.")
    
    merged['city'] = city
    print(f"  Final dataset: {len(merged)} units with income, tree size, and LST data.")
    
    return merged

# ============================================================================
# PLOTTING
# ============================================================================

def plot_tree_size_vs_income(all_data):
    """Generate the Nature-standard combined scatter plot, matching city_tree_analysis.py."""
    import matplotlib.ticker as ticker
    from scipy import stats as _stats
    from statsmodels.stats.multitest import multipletests
    try:
        import seaborn as sns
    except ImportError:
        sns = None
    from matplotlib import gridspec
    
    # 0. Preparation: Calculate p-values for BH adjustment
    p_values_raw = []
    city_order = ['austin', 'bloomington', 'cupertino', 'surrey']
    valid_cities = []
    
    for city in city_order:
        city_df = all_data[all_data['city'] == city]
        if city_df.empty: continue
        x, y = city_df['median_income'].values, city_df['median_crown_area_m2'].values
        mask = (x > 0) & (y >= 0)
        x, y = x[mask], y[mask]
        if len(x) < 5: continue
        _, p = _stats.spearmanr(x, y)
        p_values_raw.append(p)
        valid_cities.append(city)
    
    # Apply Benjamini-Hochberg correction
    if p_values_raw:
        reject, p_adjusted, _, _ = multipletests(p_values_raw, alpha=0.05, method='fdr_bh')
        bh_results = dict(zip(valid_cities, p_adjusted))
    else:
        bh_results = {}

    # helper for significance stars
    def _fmt_val(v):
        if v == 0: return "0"
        return f"{v:.2g}"

    def _fmt_sig(p):
        if p < 0.001: return '***'
        if p < 0.01:  return '**'
        if p < 0.05:  return '*'
        return ' ns'

    # Okabe-Ito palette (adjusted for grayscale legibility: Nature Cities standard)
    # Surrey (pink) and Cupertino (green) are distinguishable by lightness
    colors = {
        'austin':      '#E69F00', # Orange
        'bloomington': '#56B4E9', # Sky Blue
        'cupertino':   '#009E73', # Bluish Green
        'surrey':      '#CC79A7'  # Warm Pink/Mauve (Nature Version)
    }
    
    # Full labels to match city_tree_analysis.py
    labels = {
        'austin':      'Austin, TX',
        'bloomington': 'Bloomington, IN',
        'cupertino':   'Cupertino, CA',
        'surrey':      'Surrey, BC'
    }

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica Neue', 'Helvetica'],
        'pdf.fonttype': 42
    })

    fig = plt.figure(figsize=(7.2, 5.2))
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                           hspace=0.06, wspace=0.06)
    
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    
    legend_handles, legend_labels = [], []
    
    for city in ['austin', 'bloomington', 'cupertino', 'surrey']:
        city_df = all_data[all_data['city'] == city]
        if city_df.empty: continue
        
        # Data preparation: UNIT OF ANALYSIS confirmed as Census Unit
        x = city_df['median_income'].values
        y = city_df['median_crown_area_m2'].values
        
        # Filter for validity (include zeros for size gap analysis)
        mask = (x > 0) & (y >= 0)
        x, y = x[mask], y[mask]
        if len(x) < 5: continue
        
        # Stats: Spearman
        rho, p_raw = _stats.spearmanr(x, y)
        p_adj = bh_results.get(city, p_raw)
        stars = _fmt_sig(p_adj)
        leg_str = f"{labels[city]} $\\rho = {_fmt_val(rho)}${stars} ($p_{{adj}} = {_fmt_val(p_adj)}, n = {len(x)}$)"
        
        color = colors[city]
        
        # Transform Y and add jitter for zeros
        y_trans = np.log10(y + 1.0)
        zero_mask = (y == 0)
        if np.any(zero_mask):
            y_trans[zero_mask] += np.random.uniform(-0.015, 0.015, size=np.sum(zero_mask))
            
        # Scatter (matching city_tree_analysis.py)
        path = ax_main.scatter(x, y_trans, s=15, alpha=0.55, color=color, 
                                edgecolors='white', linewidths=0.2, zorder=3)
        legend_handles.append(path)
        legend_labels.append(leg_str)
        
        # Marginals (bandwidth unified)
        if sns is not None:
            sns.kdeplot(x=x, ax=ax_top, color=color, fill=True, alpha=0.25, lw=0, bw_adjust=1.0)
            sns.kdeplot(x=x, ax=ax_top, color=color, fill=False, alpha=0.9, lw=1.0, bw_adjust=1.0)
            sns.kdeplot(y=y_trans, ax=ax_right, color=color, fill=True, alpha=0.25, lw=0, bw_adjust=1.0)
            sns.kdeplot(y=y_trans, ax=ax_right, color=color, fill=False, alpha=0.9, lw=1.0, bw_adjust=1.0)
        
        # OLS fit on log scale (matching city_tree_analysis.py)
        try:
            x_eval = np.linspace(x.min(), x.max(), 100)
            y_log = np.log10(y + 1.0) # Unified log10(y+1)
            slope, intercept = np.polyfit(x, y_log, 1)
            z_eval = intercept + slope * x_eval
            ax_main.plot(x_eval, z_eval, color=color, lw=2.0, alpha=1.0, zorder=15)
            
            # Bootstrap for 95% CI (1000 iterations for precision)
            boots_y_trans = []
            for _ in range(1000):
                idx_b = np.random.choice(len(x), len(x), replace=True)
                xb, yb = x[idx_b], y[idx_b]
                try:
                    sb, ib = np.polyfit(xb, np.log10(yb + 1.0), 1)
                    boots_y_trans.append(ib + sb * x_eval)
                except: continue
            
            if boots_y_trans:
                boots_y_trans = np.array(boots_y_trans)
                lower_y = np.percentile(boots_y_trans, 2.5, axis=0)
                upper_y = np.percentile(boots_y_trans, 97.5, axis=0)
                ax_main.fill_between(x_eval, lower_y, upper_y, color=color, 
                                   alpha=0.20, lw=0, zorder=14)
        except Exception:
            pass

    # Clean marginals
    for sax in [ax_top, ax_right]:
        sax.axis('off')

    # Main Axis Styling
    ax_main.set_xlim(0, 250000)
    # y range in log10(y+1) space. Max 500 m2 is reasonable for median.
    ax_main.set_ylim(0, np.log10(501)) 
    
    # Tick positions in log10(y+1) space
    y_ticks_raw = [0, 10, 50, 100, 250, 500]
    ax_main.set_yticks(np.log10(np.array(y_ticks_raw) + 1))
    ax_main.set_yticklabels(['0', '10', '50', '100', '250', '500'])

    ax_main.set_xlabel(r"Median Household Income ($\times$\$1,000)", fontsize=10, fontweight='bold')
    ax_main.set_ylabel("Median Tree Crown Area (m\u00B2, log10(y+1) scale)", fontsize=10, fontweight='bold')
    
    ax_main.tick_params(axis='both', which='major', labelsize=8.5)
    ax_main.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'${int(x/1000):,}k'))
    
    ax_main.grid(False)
    ax_main.spines['right'].set_visible(False)
    ax_main.spines['top'].set_visible(False)
    
    # Legend at bottom right as requested earlier
    ax_main.legend(legend_handles, legend_labels, frameon=False, loc='lower right', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "tree_size_vs_income_combined.png"), dpi=600)
    plt.savefig(os.path.join(OUTPUT_DIR, "tree_size_vs_income_combined.pdf"))
    print(f"\nSaved publication-quality scatter plot to {OUTPUT_DIR}")

def plot_ridgeline_distribution(df):
    """Create a refined 5-panel ridgeline plot with layout and data fixes."""
    from scipy.stats import gaussian_kde, spearmanr
    from statsmodels.stats.multitest import multipletests
    import matplotlib.colors as mcolors
    from matplotlib.lines import Line2D
    import csv

    # 0. BH Adjustment for ridgeline annotations
    cities = ['austin', 'bloomington', 'cupertino', 'surrey']
    p_raw_list = []
    valid_cities = []
    for city in cities:
        city_df = df[df['city'] == city]
        if city_df.empty: continue
        x, y = city_df['median_income'], city_df['crown_area_per_ha']
        _, p = spearmanr(x, y)
        p_raw_list.append(p)
        valid_cities.append(city)
    
    bh_results = {}
    if p_raw_list:
        # Filter out NaN values (e.g. from n=1)
        valid_indices = [idx for idx, p in enumerate(p_raw_list) if not np.isnan(p)]
        if valid_indices:
            valid_p_raw = [p_raw_list[idx] for idx in valid_indices]
            valid_cities_filtered = [valid_cities[idx] for idx in valid_indices]
            
            _, p_adj, _, _ = multipletests(valid_p_raw, alpha=0.05, method='fdr_bh')
            bh_results = dict(zip(valid_cities_filtered, p_adj))

    print("\n--- DATA DIAGNOSTICS & TERTILE BREAKPOINTS ---")
    tertile_breakpoints = []
    for city in cities:
        city_df = df[df['city'] == city]
        if not city_df.empty:
            vals = city_df['crown_area_per_ha']
            print(f"{city.upper()}: count={len(vals)}, min={vals.min():.2f}, max={vals.max():.2f}, mean={vals.mean():.2f}, nulls={vals.isna().sum()}")
        else:
            print(f"{city.upper()}: EMPTY")
    print("------------------------\n")

    # 1. Standardize x-axis range: 0–150 m²/ha, capped at 95th percentile across all
    GLOBAL_PERCENTILE_95 = df['crown_area_per_ha'].quantile(0.95)
    X_MIN = 0
    X_MAX = min(150, GLOBAL_PERCENTILE_95)
    x_eval = np.linspace(X_MIN, X_MAX, 500)

    # Compute tertiles per city independently
    try:
        # We'll calculate breakpoints manually to ensure they are independent and for export
        for city in cities:
            city_df = df[df['city'] == city]
            if city_df.empty: continue
            
            incomes = city_df['median_income'].dropna()
            if len(incomes) < 3: 
                print(f"  Warning: {city} has too few data points for tertiles.")
                continue
                
            # Independent tertile breakpoints for this city
            b1, b2 = np.percentile(incomes, [33.33, 66.67])
            b_max = incomes.max()
            tertile_breakpoints.append({
                'City': city.title(),
                'Low_Max': f"{b1:.1f}",
                'Middle_Max': f"{b2:.1f}",
                'High_Max': f"{b_max:.1f}"
            })
            
            # Apply to the dataframe
            def get_tertile(val):
                if val <= b1: return 1
                if val <= b2: return 2
                return 3
            
            df.loc[df['city'] == city, 'tertile'] = city_df['median_income'].apply(get_tertile)
            
        # Export breakpoints
        bp_path = os.path.join(OUTPUT_DIR, "tertile_income_breakpoints.csv")
        with open(bp_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['City', 'Low_Max', 'Middle_Max', 'High_Max'])
            writer.writeheader()
            writer.writerows(tertile_breakpoints)
        print(f"  Exported tertile breakpoints to {bp_path}")

    except Exception as e:
        print(f"  Warning: Error computing tertiles: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. Set up figure: 180 mm width for Nature Cities
    # 7.086 inches = 180 mm
    fig = plt.figure(figsize=(7.086, 3.8), facecolor='#FFFFFF') 
    gs = plt.GridSpec(1, 5, wspace=0.18, width_ratios=[1, 1, 1, 1, 1.4])
    axes = [fig.add_subplot(gs[0, i], facecolor='#FFFFFF') for i in range(5)]
    
    city_names = ['Austin, TX', 'Bloomington, IN', 'Cupertino, CA', 'Surrey, BC']
    # Redefine colors locally
    city_colors = {'austin': '#E69F00', 'bloomington': '#56B4E9', 'cupertino': '#009E73', 'surrey': '#CC79A7'}
    
    # Typography: Arial or Helvetica
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'svg.fonttype': 'none',
        'pdf.fonttype': 42,
        'axes.unicode_minus': False
    })

    summary_data = []

    for i, city in enumerate(cities):
        ax = axes[i]
        city_df = df[df['city'] == city]
        base_color = city_colors.get(city, '#7F7F7F')
        
        if city_df.empty:
            ax.set_title(city_names[i], fontsize=8.5, fontweight='bold', color='#000000')
            continue
            
        city_median = city_df['crown_area_per_ha'].median()
        
        # Calculate Spearman rho and p-value (BH adjusted)
        rho, p_raw = spearmanr(city_df['median_income'], city_df['crown_area_per_ha'])
        p_adj = bh_results.get(city, p_raw)
        is_sig = p_adj < 0.05
        
        # Get max density across all tertiles to set offset
        max_dens = 0
        kde_data = {}
        for d in range(1, 4):
            subset = city_df[city_df['tertile'] == d]['crown_area_per_ha'].dropna()
            if len(subset) >= 1:
                try:
                    points_for_kde = len(subset)
                    if points_for_kde > 1:
                        kde = gaussian_kde(subset)
                        densities = kde(x_eval)
                    else:
                        densities = np.zeros_like(x_eval)
                        idx = np.argmin(np.abs(x_eval - subset.iloc[0]))
                        densities[max(0, idx-5):min(500, idx+5)] = 0.1
                    
                    max_dens = max(max_dens, densities.max())
                    kde_data[d] = (densities, subset.median())
                except Exception as e:
                    print(f"  KDE failed for {city} tertile {d}: {e}")
        
        # Increased offset for tertiles padding
        offset = 0.75 * max_dens * 1.6 if max_dens > 0 else 1.0
        
        # Plot tertiles
        medians_x = []
        medians_y = []
        low_med, high_med = None, None
        
        available_tertiles = sorted(kde_data.keys())
        if available_tertiles:
            for d in available_tertiles:
                _, m = kde_data[d]
                if d == 1: low_med = m
                if d == 3: high_med = m

        tertile_labels = {1: "Low Income", 2: "Middle Income", 3: "High Income"}
        for d in range(1, 4):
            if d not in kde_data: continue
            
            densities, tertile_median = kde_data[d]
            y_base = (d - 1) * offset
            y_vals = densities + y_base
            
            # Opacity/Lightness gradient for Surrey to improve separation
            alpha = 0.7
            if city == 'surrey':
                alpha_map = {1: 0.6, 2: 0.8, 3: 1.0}
                alpha = alpha_map[d]

            ax.fill_between(x_eval, y_base, y_vals, color=base_color, alpha=alpha, zorder=10-d)
            ax.plot(x_eval, y_vals, color='black', lw=0.5, alpha=0.6, zorder=20-d)
            ax.axhline(y_base, color='black', lw=0.3, alpha=0.2, zorder=5)
            
            # Tertile median dot - ensure it's plotted for every tertile
            try:
                # Find exact density at median for placement
                idx = np.argmin(np.abs(x_eval - tertile_median))
                y_dot = y_vals[idx]
                ax.scatter(tertile_median, y_dot, color='black', s=8, zorder=35)
            except Exception:
                # Fallback to y_base if KDE data is weird
                ax.scatter(tertile_median, y_base, color='black', s=8, zorder=35)
            
            medians_x.append(tertile_median)
            medians_y.append(y_dot)
            
            # Labels for Y-axis (Low/Mid/High) - No Bold
            if i == 0:
                ax.text(X_MIN - 5, y_base, tertile_labels[d], ha='right', va='center', fontsize=7.2, fontweight='normal')

        if medians_x and city != 'austin':
            # Standardize connector lines: thin solid light gray for all cities except Austin
            ax.plot(medians_x, medians_y, color='#CCCCCC', lw=0.5, alpha=0.8, zorder=25)

        if low_med is not None and high_med is not None:
            summary_data.append({'city': city, 'low': low_med, 'high': high_med, 'color': base_color, 'name': city.title()})
            print(f"{city.upper()} DATA INTEGRITY: Low Tertile med={low_med:.2f}, High Tertile med={high_med:.2f}, Delta={high_med-low_med:+.2f}")

        # Final y-limit adjustment to ensure space for titles/annotations
        ax.set_ylim(-offset*0.2, ax.get_ylim()[1] * 1.15)

        # Reference Line: City median
        ax.axvline(city_median, color='#000000', linestyle='--', lw=0.8, alpha=0.5, zorder=40)
        # Median value labels removed as requested; dashed line communicates tendency.

        # Separate Spearman rho and adjusted p-value (top-left, no box, no bold)
        stats_text = f"$\\rho$ = {rho:.2f}\n$p_{{adj}}$ {'< 0.05' if is_sig else '= ns'}"
        ax.text(0.05, 0.96, stats_text, transform=ax.transAxes, ha='left', va='top', 
                fontsize=7.0, linespacing=1.2, color='#000000', fontweight='normal')

        ax.set_title(city_names[i], fontsize=8.5, fontweight='bold', pad=20, color='#000000')
        ax.set_yticks([])
        ax.set_xlim(X_MIN, X_MAX)
        ax.tick_params(axis='x', labelsize=7.5)
        for spine in ['left', 'right', 'top']:
            ax.spines[spine].set_visible(False)

    # 5th Panel: Canopy Inequality (Dumbbell Plot)
    ax_sum = axes[4]
    ax_sum.set_title("Canopy Inequality", fontsize=8.5, fontweight='bold', pad=32, color='#000000')
    
    # Dynamic Canopy Inequality x-axis to ensure all dumbbells (including Surrey/Bloomington) are visible
    all_vals = []
    for s in summary_data:
        all_vals.extend([s['low'], s['high']])
    if all_vals:
        iq_min, iq_max = min(all_vals), max(all_vals)
        iq_margin = (iq_max - iq_min) * 0.15 if iq_max > iq_min else 10
        INEQUALITY_X_MIN, INEQUALITY_X_MAX = iq_min - iq_margin, iq_max + iq_margin
    else:
        INEQUALITY_X_MIN, INEQUALITY_X_MAX = 0, 100

    ax_sum.set_xlim(INEQUALITY_X_MIN, INEQUALITY_X_MAX)
    ax_sum.set_ylim(-0.8, 4.2)
    ax_sum.set_xlabel("", fontsize=7.5) 
    ax_sum.tick_params(axis='x', labelsize=7.0, colors='#000000')
    
    # Fixed vertical slots for the 4 cities to ensure consistent positioning
    city_slots = {
        'surrey': 3.6,
        'cupertino': 2.4,
        'bloomington': 1.2,
        'austin': 0.0
    }
    
    for s in summary_data:
        city_key = s['city'].lower()
        if city_key not in city_slots: continue
        y_idx = city_slots[city_key]
        
        # Connection line - thicker and gray
        ax_sum.plot([s['low'], s['high']], [y_idx, y_idx], color='#BBBBBB', lw=1.5, alpha=0.6, zorder=1)
        
        # Dumbbells: Open circle for Low, Filled for High - City Specific Colors
        ax_sum.scatter(s['low'], y_idx, facecolors='none', edgecolors=s['color'], s=50, lw=1.5, zorder=3)
        ax_sum.scatter(s['high'], y_idx, color=s['color'], s=50, zorder=3)
        
        delta = s['high'] - s['low']
        prefix = r"$\approx$" if abs(delta) < 0.2 else "="
        
        # Position Delta label to the right of the dumbbell
        text_x = max(s['low'], s['high']) + 2.5
        ax_sum.text(text_x, y_idx, fr"$\Delta$ {prefix} {delta:+.1f}", 
                    va='center', ha='left', fontsize=7.0, fontweight='normal', color='#000000')
        
        # City labels - Standardized positioning to the RIGHT of the dumbbells
        label_x_pos = INEQUALITY_X_MAX - (INEQUALITY_X_MAX - INEQUALITY_X_MIN) * 0.05
        ax_sum.text(label_x_pos, y_idx + 0.35, s['name'], ha='right', va='center', 
                    fontsize=7.2, fontweight='normal', color='#000000')

    # Legend for Income Tiers - clean and consolidated
    legend_elements = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor='none', markeredgecolor='#444444', 
               label='Low Income Tertile', markersize=5, markeredgewidth=1.2),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='#444444', markeredgecolor='#444444',
               label='High Income Tertile', markersize=5)
    ]
    ax_sum.legend(handles=legend_elements, loc='lower center', fontsize=7.0, frameon=False, 
                  bbox_to_anchor=(0.5, -0.45), title="Income Tiers", title_fontsize=7.2, 
                  handletextpad=0.6, labelspacing=0.8, ncol=1)

    ax_sum.set_yticks([])
    for spine in ['left', 'right', 'top']:
        ax_sum.spines[spine].set_visible(False)
    ax_sum.spines['bottom'].set_visible(True)
    ax_sum.spines['bottom'].set_color('#000000')

    ax_sum.set_yticks([])
    for spine in ['left', 'right', 'top']:
        ax_sum.spines[spine].set_visible(False)

    fig.text(0.5, 0.04, "Tree Canopy Cover (m\u00B2/ha)", ha='center', fontsize=8, fontweight='bold')
    
    # Manual Adjust for 180mm width and non-overlapping elements
    # Increased wspace to 0.35 to prevent label bleed between panels
    plt.subplots_adjust(left=0.06, right=0.98, top=0.85, bottom=0.25, wspace=0.35)
    
    # Final Export: 180 mm width (7.086 in), 300 dpi minimum
    plt.savefig(os.path.join(OUTPUT_DIR, "figure2_ridgeline.png"), dpi=600)
    plt.savefig(os.path.join(OUTPUT_DIR, "figure2_ridgeline.pdf"), format='pdf', transparent=False)
    plt.savefig(os.path.join(OUTPUT_DIR, "figure2_ridgeline.tif"), dpi=300, format='tiff', pil_kwargs={"compression": "tiff_lzw"})
    
    print(f"\n  Saved final Nature Cities figures to {OUTPUT_DIR}")
    print(f"\n  Saved refined 5-panel ridgeline plot to {OUTPUT_DIR}")

def plot_mediation_diagrams(mediation_results):
    """Create a 2x2 panel mediation path diagram figure."""
    import matplotlib.patches as patches
    
    city_order = ['austin', 'bloomington', 'cupertino', 'surrey']
    city_names = {'austin': 'Austin, TX', 'bloomington': 'Bloomington, IN', 
                  'cupertino': 'Cupertino, CA', 'surrey': 'Surrey, BC'}
    city_colors = {'austin': '#E69F00', 'bloomington': '#56B4E9', 
                   'cupertino': '#009E73', 'surrey': '#CC79A7'}
    
    fig, axes = plt.subplots(2, 2, figsize=(7.086, 6), facecolor='white')
    axes = axes.flatten()
    
    for i, city in enumerate(city_order):
        ax = axes[i]
        res = next((r for r in mediation_results if r['City'].lower() == city), None)
        if not res: 
            ax.axis('off')
            continue
        
        # Positions: Income left, Trees top, LST right
        pos = {'Income': (0.15, 0.35), 'Trees': (0.5, 0.75), 'LST': (0.85, 0.35)}
        
        def draw_node(p, label, color):
            ax.add_patch(patches.FancyBboxPatch((p[0]-0.08, p[1]-0.05), 0.16, 0.1, 
                                                boxstyle="round,pad=0.03", color='white', 
                                                ec=color, lw=1.5, zorder=10))
            ax.text(p[0], p[1], label, ha='center', va='center', fontsize=8, fontweight='bold', zorder=11)

        draw_node(pos['Income'], 'Income', '#444444')
        draw_node(pos['Trees'], 'Tree\nDensity', city_colors[city])
        draw_node(pos['LST'], 'Summer\nLST', '#D55E00')
        
        # Arrows
        def draw_arrow(p1, p2, label, sig, va='bottom', ha='center', label_pos=0.5):
            color = city_colors[city] if sig else '#999999'
            ax.annotate("", xy=p2, xytext=p1, arrowprops=dict(arrowstyle="->", color=color, lw=2, shrinkA=18, shrinkB=18))
            text_x = p1[0] + (p2[0]-p1[0])*label_pos
            text_y = p1[1] + (p2[1]-p1[1])*label_pos
            ax.text(text_x, text_y, label, ha=ha, va=va, fontsize=8, color=color, fontweight='bold' if sig else 'normal')

        draw_arrow(pos['Income'], pos['Trees'], f"a={res['a']:.2f}", res['p_a'] < 0.05, va='bottom', ha='right', label_pos=0.5)
        draw_arrow(pos['Trees'], pos['LST'], f"b={res['b']:.2f}", res['p_b'] < 0.05, va='bottom', ha='left', label_pos=0.5)
        draw_arrow(pos['Income'], pos['LST'], f"c'={res['c_prime']:.2f}", res['p_direct'] < 0.05, va='top', label_pos=0.5)
        
        # Indirect Effect Label
        color = 'green' if res['significant'] else '#666666'
        sig_str = "*" if res['significant'] else ""
        indirect_text = f"Indirect Effect: {res['Indirect_Effect']:.3f}{sig_str}\n95% CI [{res['Indirect_CI_Lower']:.3f}, {res['Indirect_CI_Upper']:.3f}]"
        ax.text(0.5, 0.12, indirect_text, ha='center', va='center', fontsize=7.5, color=color, 
                bbox=dict(facecolor='white', edgecolor=color, alpha=0.1, boxstyle='round,pad=0.3'))
        
        ax.set_title(city_names[city], fontsize=10, fontweight='bold', pad=5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "figure_mediation.pdf"), format='pdf', dpi=300)
    plt.savefig(os.path.join(OUTPUT_DIR, "figure_mediation.tiff"), dpi=300, format='tiff')
    plt.close()


def _add_scale_bar(ax, bounds):
    """White-background scale bar in the bottom-right corner."""
    bx0, by0, bx1, by1 = bounds
    mid_lat = (by0 + by1) / 2.0
    km_per_deg = 111.0 * cos(radians(mid_lat))
    span_km = (bx1 - bx0) * km_per_deg
    target = span_km * 0.20
    bar_km = 1
    for bk in [1, 2, 5, 10, 20, 50, 100]:
        if bk >= target * 0.6:
            bar_km = bk
            break
    bar_deg = bar_km / km_per_deg
    pad_x = (bx1 - bx0) * 0.06
    pad_y = (by1 - by0) * 0.08
    x0 = bx1 - pad_x - bar_deg
    y0 = by0 + pad_y
    # Background patch
    ax.add_patch(patches.Rectangle(
        (x0 - bar_deg * 0.15, y0 - pad_y * 0.6),
        bar_deg * 1.35, pad_y * 2.0,
        facecolor='white', edgecolor='none', alpha=0.85, zorder=8))
    ax.plot([x0, x0 + bar_deg], [y0, y0],
            color='black', lw=1.2, solid_capstyle='butt', zorder=9)
    tick_h = (by1 - by0) * 0.012
    ax.plot([x0, x0], [y0 - tick_h, y0 + tick_h],
            color='black', lw=0.7, zorder=9)
    ax.plot([x0 + bar_deg, x0 + bar_deg],
            [y0 - tick_h, y0 + tick_h],
            color='black', lw=0.7, zorder=9)
    ax.text(x0 + bar_deg / 2.0, y0 + tick_h * 2.0,
            f'{bar_km} km', ha='center', va='bottom',
            fontsize=7, color='black', zorder=9)

def plot_bivariate_maps(df_full):
    """Create 2x2 bivariate choropleth maps with summary vulnerability chart."""
    cities = ['austin', 'bloomington', 'cupertino', 'surrey']
    city_names = {'austin': 'Austin, TX', 'bloomington': 'Bloomington, IN', 
                  'cupertino': 'Cupertino, CA', 'surrey': 'Surrey, BC'}
    city_colors = {'austin': '#E69F00', 'bloomington': '#56B4E9', 
                   'cupertino': '#009E73', 'surrey': '#CC79A7'}
    
    # 1. Standardize Colormap: Global tertiles for cross-city consistency
    df_full = df_full.copy()
    df_full['canopy_tertile'] = pd.qcut(df_full['tree_density_km2'], 3, labels=[1, 2, 3])
    df_full['lst_tertile'] = pd.qcut(df_full['mean_lst'], 3, labels=[1, 2, 3])
    
    # Viridis 3x3 Palette (Row=LST, Col=Canopy)
    biv_palette = {
        (1, 1): '#fde725', (2, 1): '#99cf4f', (3, 1): '#35b779',
        (1, 2): '#e08b58', (2, 2): '#85735c', (3, 2): '#2a5c60',
        (1, 3): '#c42f8a', (2, 3): '#711869', (3, 3): '#1e0048',
    }
    
    # Output figure size: Nature standard sizing
    fig = plt.figure(figsize=(10, 10), facecolor='white')
    # 2 rows for maps (0,1), 1 row for legend/chart (2)
    gs = plt.GridSpec(3, 2, height_ratios=[1, 1, 0.4])
    
    # Figure title and subtitle (Replicating city_tree_analysis.py structure)
    fig.suptitle('Income \u00b7 Canopy \u00b7 Heat: Bivariate Equity Analysis', 
                 fontsize=14, fontweight='bold', y=0.98)
    fig.text(0.5, 0.95, 
             'U.S. Census Block Groups \u00b7 Canada: Dissemination Areas', 
             ha='center', va='top', fontsize=9, color='#666666', style='italic')

    vuln_stats = []
    
    official_boundaries = {c: _load_place_boundary(c) for c in cities}

    for i, city in enumerate(cities):
        ax = fig.add_subplot(gs[i//2, i%2])
        city_df = df_full[df_full['city'] == city].copy()
        if city_df.empty: 
            ax.axis('off')
            continue
            
        # Merge with shapefile for mapping
        shp_path = CENSUS_BOUNDARY_PATHS.get(city)
        if not shp_path or not os.path.exists(shp_path):
            ax.axis('off')
            continue
            
        gdf = gpd.read_file(shp_path)
        if gdf.crs != 'EPSG:4326': gdf = gdf.to_crs('EPSG:4326')
        geoid_col = 'DAUID' if city == 'surrey' else 'GEOID'
        
        map_gdf = gdf.merge(city_df, left_on=geoid_col, right_on='GEOID', how='inner')
        map_gdf['biv_color'] = map_gdf.apply(lambda r: biv_palette[(int(r.canopy_tertile), int(r.lst_tertile))], axis=1)
        
        # Clip to city boundary
        boundary = official_boundaries.get(city)
        map_bounds = None
        if boundary and not boundary.is_empty:
            map_gdf['geometry'] = map_gdf.geometry.intersection(boundary)
            map_gdf = map_gdf[~map_gdf.is_empty]
            # Thick black city boundary outline (Replicating city_tree_analysis.py)
            gpd.GeoSeries([boundary]).plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.2, zorder=10)
            cb = boundary.bounds
            p_factor = 0.04 if (cb[2] - cb[0]) > 0.1 else 0.06
            px = (cb[2] - cb[0]) * p_factor
            py = (cb[3] - cb[1]) * p_factor
            map_bounds = (cb[0] - px, cb[1] - py, cb[2] + px, cb[3] + py)
        
        # Plot base
        map_gdf.plot(color=map_gdf['biv_color'], ax=ax, edgecolor='white', linewidth=0.1)
        
        # Highlight low-income tertile
        low_inc_b = city_df['median_income'].quantile(0.333)
        low_inc_gdf = map_gdf[map_gdf['median_income'] <= low_inc_b]
        low_inc_gdf.plot(ax=ax, facecolor='none', edgecolor='#444444', linewidth=0.5, alpha=0.7)
        
        # Summary calculation for vulnerability bar chart
        total_low_inc = len(low_inc_gdf)
        vuln_low_inc = len(low_inc_gdf[(low_inc_gdf['canopy_tertile'] == 1) & (low_inc_gdf['lst_tertile'] == 3)])
        perc_vuln = (vuln_low_inc / total_low_inc * 100) if total_low_inc > 0 else 0
        vuln_stats.append({'City': city.title(), 'Perc': perc_vuln, 'Color': city_colors[city]})
        
        # Set viewport and North-up orientation
        ax.set_aspect('equal')
        if map_bounds:
            ax.set_xlim(map_bounds[0], map_bounds[2])
            ax.set_ylim(map_bounds[1], map_bounds[3])
        else:
            # Fallback if no official boundary
            minx, miny, maxx, maxy = map_gdf.total_bounds
            pad_x, pad_y = (maxx - minx) * 0.05, (maxy - miny) * 0.05
            ax.set_xlim(minx - pad_x, maxx + pad_x)
            ax.set_ylim(miny - pad_y, maxy + pad_y)
            map_bounds = (ax.get_xlim()[0], ax.get_ylim()[0], ax.get_xlim()[1], ax.get_ylim()[1])

        # Panel Annotations (Replicating city_tree_analysis.py)
        ax.set_title(city_names[city], fontsize=10, fontweight='bold', pad=8)
        
        n_trees = int(city_df['tree_count'].sum())
        n_units = len(city_df)
        geo_unit = "Dissemination Areas" if city == 'surrey' else "Block Groups"
        panel_subtitle = f"$n$ = {n_trees:,} trees across {n_units:,} {geo_unit}"
        
        ax.text(0.5, -0.05, panel_subtitle,
                transform=ax.transAxes, fontsize=7.5,
                color='#444444', style='italic', va='top', ha='center',
                bbox={'facecolor': 'white', 'alpha': 0.9, 'edgecolor': 'none', 'pad': 1.0},
                zorder=15)
        
        # Add scale bar
        if map_bounds:
            _add_scale_bar(ax, map_bounds)
        
        ax.axis('off')

    # Bivariate Legend: Left column of row 2 (bottom row)
    ax_leg = fig.add_subplot(gs[2, 0])
    ax_leg.axis('off')
    bs = 0.15 # Box size
    start_x, start_y = 0.35, 0.2
    for (c, r), color in biv_palette.items():
        rect = patches.Rectangle((start_x + (c-1)*bs, start_y + (r-1)*bs), bs, bs, 
                                 color=color, transform=ax_leg.transAxes)
        ax_leg.add_patch(rect)
    
    ax_leg.text(start_x + 1.5*bs, start_y - bs*0.2, "Tree Density \u2192", 
                transform=ax_leg.transAxes, fontsize=10, ha='center', va='top')
    ax_leg.text(start_x - bs*0.2, start_y + 1.5*bs, "Summer LST \u2192", 
                transform=ax_leg.transAxes, fontsize=10, va='center', rotation=90, ha='right')

    # Vulnerability Bar Chart: Right column of row 2 (bottom row)
    ax_bar = fig.add_subplot(gs[2, 1])
    v_stats_df = pd.DataFrame(vuln_stats)
    if not v_stats_df.empty:
        v_stats_df['City'] = pd.Categorical(v_stats_df['City'], categories=[city_names[c].split(',')[0] for c in cities], ordered=True)
        v_stats_df = v_stats_df.sort_values('City')
        
        bars = ax_bar.bar(v_stats_df['City'], v_stats_df['Perc'], color=v_stats_df['Color'], alpha=0.9, width=0.6)
        ax_bar.set_ylabel("% Low-Income Units in\nLow Canopy/High Heat", fontsize=10, fontweight='bold', labelpad=10)
    
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.yaxis.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    # Save high-res output
    out_png = os.path.join(OUTPUT_DIR, "figure_bivariate_map.png")
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_png.replace('.png', '.pdf'), format='pdf', dpi=300)
    plt.close()
    print(f"Revised figure saved to {out_png} at 300dpi")

# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    cities = ['austin', 'bloomington', 'cupertino', 'surrey']
    all_results = []
    
    for city in cities:
        res = run_tree_size_analysis(city, CUTOUTS_DIR)
        if res is not None:
            all_results.append(res)
            
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        
        # Standardize Income and LST within each city independently
        # Log-transform tree density (trees/km2) and standardize
        combined['tree_density_log'] = np.log1p(combined['tree_density_km2'])
        
        for city in cities:
            mask = combined['city'] == city
            if not combined[mask].empty:
                # Log-transform tree density to address right skew
                combined.loc[mask, 'income_std'] = (combined.loc[mask, 'median_income'] - combined.loc[mask, 'median_income'].mean()) / combined.loc[mask, 'median_income'].std()
                combined.loc[mask, 'lst_std'] = (combined.loc[mask, 'mean_lst'] - combined.loc[mask, 'mean_lst'].mean()) / combined.loc[mask, 'mean_lst'].std()
                combined.loc[mask, 'tree_density_log_std'] = (combined.loc[mask, 'tree_density_log'] - combined.loc[mask, 'tree_density_log'].mean()) / combined.loc[mask, 'tree_density_log'].std()
        
        combined.to_csv(os.path.join(OUTPUT_DIR, "census_units_full.csv"), index=False)
        
        # 1. Mediation Analysis
        mediation_results = []
        for city in cities:
            city_df = combined[combined['city'] == city.lower()].dropna(subset=['income_std', 'lst_std', 'tree_density_log_std'])
            if len(city_df) >= 5: # Threshold lowered for sample data
                mediation_results.append(run_mediation(city_df, city.title()))
        
        # Pooled Model with City Fixed Effects
        valid_pooled = combined.dropna(subset=['income_std', 'lst_std', 'tree_density_log_std']).copy()
        # Count units per city in pooled
        pooled_counts = valid_pooled['city'].value_counts()
        cities_to_pool = pooled_counts[pooled_counts >= 5].index.tolist()
        
        if len(cities_to_pool) >= 2:
            pooled_df = valid_pooled[valid_pooled['city'].isin(cities_to_pool)].copy()
            # Add city dummies (only for cities in pooled_df, excluding first)
            # Ensure dummies are integers (not booleans) to avoid OLS object dtype errors
            dummies = pd.get_dummies(pooled_df['city'], prefix='city', drop_first=True).astype(int)
            pooled_df = pd.concat([pooled_df.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
            
            # Use columns directly to avoid dummy name issues
            X_cols = ['income_std'] + [c for c in dummies.columns]
            M_cols = ['income_std', 'tree_density_log_std'] + [c for c in dummies.columns]
            
            # Final safety: Ensure all regression columns are float
            for col in ['tree_density_log_std', 'lst_std'] + M_cols:
                pooled_df[col] = pd.to_numeric(pooled_df[col], errors='coerce')
            
            pooled_df = pooled_df.dropna(subset=['tree_density_log_std', 'lst_std'] + M_cols)
            
            ma = sm.OLS(pooled_df['tree_density_log_std'], sm.add_constant(pooled_df[X_cols])).fit()
            mbc = sm.OLS(pooled_df['lst_std'], sm.add_constant(pooled_df[M_cols])).fit()
            mt = sm.OLS(pooled_df['lst_std'], sm.add_constant(pooled_df[X_cols])).fit()
            
            a, b, c_prime, total = ma.params['income_std'], mbc.params['tree_density_log_std'], mbc.params['income_std'], mt.params['income_std']
            mediation_results.append({
                'City': 'Pooled', 'a': a, 'p_a': ma.pvalues['income_std'],
                'b': b, 'p_b': mbc.pvalues['tree_density_log_std'],
                'c_prime': c_prime, 'p_c_prime': mbc.pvalues['income_std'],
                'Total_Effect': total, 'Direct_Effect': c_prime,
                'Indirect_Effect': a * b, 'Indirect_CI_Lower': np.nan, 'Indirect_CI_Upper': np.nan,
                'Proportion_Mediated': (a * b) / total if total != 0 else 0, 'p_total': mt.pvalues['income_std'],
                'p_direct': mbc.pvalues['income_std'], 'significant': True
            })
            
        res_df = pd.DataFrame(mediation_results)
        res_df.to_csv(os.path.join(OUTPUT_DIR, "mediation_results.csv"), index=False)
        if not res_df.empty:
            print("\n--- MEDIATION ANALYSIS RESULTS ---")
            cols_to_show = ['City', 'Total_Effect', 'Direct_Effect', 'Indirect_Effect', 'Proportion_Mediated']
            cols_to_show = [c for c in cols_to_show if c in res_df.columns]
            if 'p_indirect' in res_df.columns: cols_to_show.append('p_indirect')
            elif 'significant' in res_df.columns: cols_to_show.append('significant')
            print(res_df[cols_to_show])
        else:
            print("\n--- NO MEDIATION RESULTS GENERATED ---")
        
        # 2. Figures
        plot_tree_size_vs_income(combined)
        plot_ridgeline_distribution(combined)
        
        if mediation_results:
            plot_mediation_diagrams(mediation_results)
            plot_bivariate_maps(combined)
        
        # Now Print Summary Table
        print("\n" + "="*95)
        print(f"{'City':<15} {'n':<6} {'rho':<8} {'LST (C)':<8} {'Tree/km2':<10} {'MedCrown':<10} {'Delta':<8}")
        print("-" * 95)
        
        for city in cities:
            city_df = combined[combined['city'] == city]
            if city_df.empty: continue
            
            n = len(city_df)
            r, p = spearmanr(city_df['median_income'], city_df['median_crown_area_m2'])
            mean_lst = city_df['mean_lst'].mean()
            mean_tree = city_df['tree_density_km2'].mean()
            med_crown = city_df['median_crown_area_m2'].median()
            
            # Delta calculation
            low = city_df[city_df['tertile'] == 1]['crown_area_per_ha'].median()
            high = city_df[city_df['tertile'] == 3]['crown_area_per_ha'].median()
            delta = high - low if (not np.isnan(low) and not np.isnan(high)) else np.nan
            
            print(f"{city.title():<15} {n:<6} {r:<8.2f} {mean_lst:<8.1f} {mean_tree:<10.0f} {med_crown:<10.1f} {delta:<+8.1f}")
        print("="*95 + "\n")
    else:
        print("\nNo data collected to plot.")
