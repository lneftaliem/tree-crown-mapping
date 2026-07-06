"""
Identical Tree Density Map Generator (Point-Based)
--------------------------------------------------
Generates the publication-quality tree density maps for Austin, Bloomington, 
Cupertino, and Surrey using pre-extracted tree point datasets (GeoJSON).

This script replicates the exact aesthetics and logic of city_tree_analysis.py
but bypasses the expensive raster-based tree detection by using the final
deduplicated tree point datasets.

Usage:
    python generate_density_map_from_geojson.py
"""

import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, Polygon as MplPolygon
from matplotlib.colors import Normalize as _MplNorm
from shapely.geometry import box, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid
from math import radians, cos
import gc

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

# Sherlock Base Path
BASE_DIR = os.environ.get("TREE_MAPPING_BASE_DIR", os.getcwd())
# Sherlock path for point datasets (Ripley CSVs)
DATA_DIR = os.path.join(BASE_DIR, "analysis_output", "ripley_data")

# Output directory for the final map
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# New Exact Boundaries Configuration
EXACT_BOUNDARIES_DIR = os.path.join(BASE_DIR, 'census_boundaries', 'exact_boundaries')

# Standardized UTM projections (meters) from prepare_ripley_data.py
CITY_UTM_EPSG = {
    'austin':      32614,   # UTM 14N
    'bloomington': 32616,   # UTM 16N
    'cupertino':   32610,   # UTM 10N
    'surrey':      32610,   # UTM 10N
}

# Standardized UTM projections (meters) from prepare_ripley_data.py
CITY_UTM_EPSG = {
    'austin':      32614,   # UTM 14N
    'bloomington': 32616,   # UTM 16N
    'cupertino':   32610,   # UTM 10N
    'surrey':      32610,   # UTM 10N
}

# Minimum area threshold for map visibility (km2)
MIN_AREA_KM2_MAP = 0.001

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_place_boundary(city_lower):
    """Load exact city limit boundary from GeoJSON."""
    path = os.path.join(EXACT_BOUNDARIES_DIR, f"{city_lower}_exact_boundary.geojson")
    if not os.path.exists(path):
        return None

    try:
        gdf = gpd.read_file(path)
        if gdf.crs and gdf.crs != 'EPSG:4326':
            gdf = gdf.to_crs('EPSG:4326')
        
        boundary = unary_union(gdf.geometry.dropna())
        boundary = make_valid(boundary) if not boundary.is_valid else boundary
        return boundary
    except Exception as e:
        print(f"Error loading boundary for {city_lower}: {e}")
        return None

def load_census_gdf(city_lower, boundary=None):
    """Load exact census block groups for a city from GeoJSON."""
    path = os.path.join(EXACT_BOUNDARIES_DIR, f"{city_lower}_exact_block_groups.geojson")
    if not os.path.exists(path):
        return None

    try:
        census_gdf = gpd.read_file(path)
        if census_gdf.crs and census_gdf.crs != 'EPSG:4326':
            census_gdf = census_gdf.to_crs('EPSG:4326')
        
        # Initial selection using bounding box to ensure internal enclaves are caught
        if boundary:
            census_gdf['geometry'] = census_gdf.geometry.map(lambda g: make_valid(g) if not g.is_valid else g)
            census_gdf = census_gdf[census_gdf.geometry.intersects(box(*boundary.bounds))]
            
        return census_gdf
    except Exception as e:
        print(f"Error loading census for {city_lower}: {e}")
        return None

# ============================================================================
# MAIN GENERATION LOGIC
# ============================================================================

def main():
    cities = ['austin', 'bloomington', 'cupertino', 'surrey']
    city_labels = {'austin': 'Austin, TX', 'bloomington': 'Bloomington, IN', 
                   'cupertino': 'Cupertino, CA', 'surrey': 'Surrey, BC'}
    
    city_results = {}
    all_densities = []

    print(f"Loading data from: {DATA_DIR}")
    
    for city in cities:
        print(f"\nProcessing {city.title()}...")
        
        # 1. Load Trees (from GeoJSON)
        geojson_path = os.path.join(DATA_DIR, f"{city}_ripley_points_with_income.geojson")
        if not os.path.exists(geojson_path):
            print(f"  Warning: {geojson_path} not found. Skipping.")
            continue
            
        trees_gdf = gpd.read_file(geojson_path)
        print(f"  Loaded {len(trees_gdf):,} trees from GeoJSON.")
        t_bounds = trees_gdf.total_bounds
        print(f"  Trees spatial extent: {t_bounds[0]:.4f}, {t_bounds[1]:.4f} to {t_bounds[2]:.4f}, {t_bounds[3]:.4f}")
        
        # 2. Load Boundaries and Coverage
        boundary = load_place_boundary(city)
        if boundary:
            b_bounds = boundary.bounds
            print(f"  City boundary extent: {b_bounds[0]:.4f}, {b_bounds[1]:.4f} to {b_bounds[2]:.4f}, {b_bounds[3]:.4f}")

        census_gdf = load_census_gdf(city, boundary)
        
        # Load Coverage Mask from tile_metadata.csv to identify analyzed areas
        coverage_mask = None
        meta_path = os.path.join(os.getcwd(), 'tile_metadata.csv')
        if os.path.exists(meta_path):
            try:
                df_meta = pd.read_csv(meta_path)
                # Filter for this city's tiles
                keywords = {'austin':['austin'], 'bloomington':['bloomington'], 
                            'cupertino':['cupertino'], 'surrey':['surrey']}.get(city, [city])
                city_meta = df_meta[df_meta['filename'].str.lower().apply(lambda fn: any(kw in fn for kw in keywords))]
                if len(city_meta) > 0:
                    tile_boxes = [box(r.left, r.bottom, r.right, r.top) for r in city_meta.itertuples()]
                    coverage_mask = unary_union(tile_boxes).buffer(0.001) # Small buffer to bridge gaps
                    print(f"  Created coverage mask from {len(city_meta)} tiles.")
            except Exception as e:
                print(f"  Warning: Could not create coverage mask: {e}")
        
        if census_gdf is None or len(census_gdf) == 0:
            print(f"  Warning: No census data found for {city}.")
            continue
            
        # 3. Spatial Join and Density Calculation
        print("  Aggregating trees to block groups...")
        # Ensure CRS match
        if trees_gdf.crs != census_gdf.crs:
            trees_gdf = trees_gdf.to_crs(census_gdf.crs)
            
        joined = gpd.sjoin(trees_gdf, census_gdf, how='inner', predicate='intersects')
        # Ensure 1:1 assignment for trees on boundaries
        joined = joined[~joined.index.duplicated(keep='first')]
        counts = joined.groupby('index_right').size()
        
        census_gdf['tree_count'] = 0
        census_gdf.loc[counts.index, 'tree_count'] = counts.values
        
        n_assigned = int(census_gdf['tree_count'].sum())
        n_outside = len(trees_gdf) - n_assigned
        
        # Calculate Area of the PART inside the city boundary (for accurate density)
        geoid_col = next((c for c in ('GEOID', 'census_id', 'DAUID', 'GEO_ID') if c in census_gdf.columns), 'index')
        
        def get_clipped_area_km2(row):
            geom = row.geometry
            if boundary:
                try:
                    # Use the same clipping logic as drawing
                    geom = make_valid(geom).intersection(make_valid(boundary))
                except: pass
            
            if geom.is_empty: return 0
            # Geodesic approx for km2
            return geom.area * 12391.0 * np.cos(np.radians(geom.centroid.y))

        census_gdf['area_km2'] = census_gdf.apply(get_clipped_area_km2, axis=1)
        census_gdf['tree_density'] = census_gdf['tree_count'] / census_gdf['area_km2']
        census_gdf.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Diagnostics for "Empty" block groups
        n_zero = 0
        n_zero_covered = 0
        zero_area_km2 = 0
        n_nan = len(census_gdf[census_gdf['tree_density'].isna()])
        sample_ids = []
        for r in census_gdf.itertuples():
            if r.tree_count == 0 and r.area_km2 > 0:
                n_zero += 1
                if coverage_mask and coverage_mask.intersects(r.geometry):
                    n_zero_covered += 1
                    zero_area_km2 += r.area_km2
                    if len(sample_ids) < 5: sample_ids.append(getattr(r, geoid_col))
        
        print(f"  Aggregated: {n_assigned:,} trees into {len(counts)}/{len(census_gdf)} block groups.")
        print(f"  Status: {len(counts)} with trees, {n_zero} with zero trees ({n_zero_covered} covered by imagery), {n_nan} with no data.")
        if n_zero_covered > 0:
            print(f"  Total area of zero-tree covered units: {zero_area_km2:.2f} km\u00b2")
            print(f"  Sample IDs with zero trees (but covered by imagery): {sample_ids}")
        
        if n_outside > 0:
            print(f"  ⚠ {n_outside:,} trees fell outside all loaded block groups.")

        # No longer filtering tiny units to ensure full map coverage
        all_densities.extend(census_gdf['tree_density'].dropna().tolist())
        
        city_results[city] = {
            'census': census_gdf,
            'boundary': boundary,
            'coverage_mask': coverage_mask,
            'count': int(census_gdf['tree_count'].sum())
        }

    # ========================================================================
    # PLOTTING
    # ========================================================================
    
    if not city_results:
        print("No data processed. Exiting.")
        return

    # shared continuous colour scale (fixed at 5000 as per original script)
    vmax_dens = 5000 
    vmin_dens = 0
    cmap_dens = plt.cm.YlGn
    norm_dens = _MplNorm(vmin=vmin_dens, vmax=vmax_dens)

    def get_color(dens):
        if pd.isna(dens):
            return '#E0E0E0' # No Data
        # Use colormap for zero to distinguish from background/missing data
        return cmap_dens(norm_dens(min(max(0, dens), vmax_dens)))

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 18,
    })

    fig = plt.figure(figsize=(10.0, 8.0))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.05], hspace=0.30, wspace=0.10)

    for idx, city in enumerate(cities):
        if city not in city_results:
            continue
            
        row, col_idx = divmod(idx, 2)
        ax = fig.add_subplot(gs[row, col_idx])
        ax.set_facecolor('white')
        
        data = city_results[city]
        census = data['census']
        boundary = data['boundary']
        city_lower = city.lower()
        
        # Draw Choropleth
        coverage_mask = data.get('coverage_mask')
        for r in census.itertuples():
            # Skip block groups with no imagery coverage to avoid "empty" holes
            if coverage_mask and not coverage_mask.intersects(r.geometry):
                continue
                
            color = get_color(r.tree_density)
            geom = r.geometry
            if boundary:
                try:
                    # Clip and ensure we only keep polygonal parts
                    geom = make_valid(geom).intersection(make_valid(boundary))
                except: continue
            
            if geom.is_empty: continue
            
            # Robustly handle Polygon, MultiPolygon, or GeometryCollection
            polys = []
            if geom.geom_type == 'Polygon':
                polys = [geom]
            elif geom.geom_type == 'MultiPolygon':
                polys = list(geom.geoms)
            elif geom.geom_type == 'GeometryCollection':
                polys = [p for p in geom.geoms if p.geom_type in ['Polygon', 'MultiPolygon']]
                # If nested MultiPolygons, flatten them
                flat_polys = []
                for p in polys:
                    if p.geom_type == 'MultiPolygon': flat_polys.extend(list(p.geoms))
                    else: flat_polys.append(p)
                polys = flat_polys

            for p in polys:
                if p.is_empty: continue
                ax.add_patch(MplPolygon(list(zip(*p.exterior.xy)), facecolor=color, edgecolor='#444444', lw=0.3, zorder=1))

        # Draw City Outline
        if boundary:
            if boundary.geom_type == 'Polygon':
                ax.plot(*boundary.exterior.xy, color='black', lw=1.2, zorder=10)
            elif boundary.geom_type == 'MultiPolygon':
                for p in boundary.geoms:
                    ax.plot(*p.exterior.xy, color='black', lw=1.2, zorder=10)

        # Title
        ax.set_title(city_labels[city], fontsize=20, fontweight='bold', pad=6)
        
        # Axis Limits (Exact match to original script zoom logic)
        if boundary:
            xmin, ymin, xmax, ymax = boundary.bounds
        else:
            xmin, ymin, xmax, ymax = census.total_bounds
            
        if city_lower == 'austin':
            centroid = boundary.centroid
            cx, cy = centroid.x, centroid.y
            span = 0.25
            ax.set_xlim(cx - span, cx + span)
            ax.set_ylim(cy - span, cy + span)
        elif city_lower == 'surrey':
            padding_x = (xmax - xmin) * 0.01
            padding_y = (ymax - ymin) * 0.01
            ax.set_xlim(xmin - padding_x, xmax + padding_x)
            ax.set_ylim(ymin - padding_y, ymax + padding_y)
        else:
            padding_x = (xmax - xmin) * 0.05
            padding_y = (ymax - ymin) * 0.05
            ax.set_xlim(xmin - padding_x, xmax + padding_x)
            ax.set_ylim(ymin - padding_y, ymax + padding_y)
            
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

        # Caption (n=...)
        n_val = data["count"]
        if n_val > 0:
            from math import log10, floor
            rounded_n = int(round(n_val, -floor(log10(n_val)) + 2))
        else:
            rounded_n = 0
        subtitle = f'$n$ = {rounded_n:,} trees'
        ax.text(0.5, -0.05, subtitle, transform=ax.transAxes, fontsize=16,
                color='#444444', style='italic', va='top', ha='center',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=1.0),
                zorder=15)

        # Scale Bar
        BAR_CONFIGS = {'austin': 10, 'bloomington': 2, 'cupertino': 2, 'surrey': 5}
        bar_km = BAR_CONFIGS.get(city_lower, 2)
        
        _xlim = ax.get_xlim()
        _ylim = ax.get_ylim()
        mid_lat = (_ylim[0] + _ylim[1]) / 2.0
        km_per_deg = 111.0 * np.cos(np.radians(mid_lat))
        bar_deg = bar_km / km_per_deg
        
        pad_x_sb = (_xlim[1] - _xlim[0]) * 0.06
        pad_y_sb = (_ylim[1] - _ylim[0]) * 0.05
        x0_sb = _xlim[1] - pad_x_sb - bar_deg
        y0_sb = _ylim[0] + pad_y_sb
        
        if city_lower == 'austin': y0_sb -= (_ylim[1] - _ylim[0]) * 0.02
        
        ax.plot([x0_sb, x0_sb + bar_deg], [y0_sb, y0_sb], color='black', lw=1.2, zorder=15)
        ax.text(x0_sb + bar_deg / 2.0, y0_sb + (_ylim[1] - _ylim[0]) * 0.02, f'{bar_km} km', 
                ha='center', va='bottom', fontsize=15, color='black', zorder=15)

    # Colorbar
    cax = fig.add_subplot(gs[:, 2])
    sm = plt.cm.ScalarMappable(cmap=cmap_dens, norm=norm_dens)
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(r'Tree Density (trees km$^{-2}$)', fontsize=18, fontweight='bold')
    
    _ticks = [0, 1000, 2000, 3000, 4000, 5000]
    cbar.set_ticks(_ticks)
    _labels = [f'{int(t):,}' for t in _ticks]
    _labels[-1] = '5,000+'
    cbar.set_ticklabels(_labels)
    cbar.ax.tick_params(labelsize=16)
    
    # Save
    out_png = os.path.join(OUTPUT_DIR, "identical_tree_density_map.png")
    out_pdf = out_png.replace(".png", ".pdf")
    plt.savefig(out_png, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.savefig(out_pdf, bbox_inches='tight', pad_inches=0.05)
    print(f"\nMap saved to: {out_png}")
    plt.close()

if __name__ == "__main__":
    main()
