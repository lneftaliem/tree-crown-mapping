import os
import sys
import gc
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Polygon as MplPolygon, Rectangle
from matplotlib.colors import Normalize as _MplNorm
from math import cos, radians

# Setup paths
BASE_DIR = os.getcwd()
CSV_PATH = os.path.join(BASE_DIR, "notebooks/urbantreedata_2023/new_data.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Census paths mapping
CENSUS_BOUNDARY_PATHS = {
    'austin': os.path.join(BASE_DIR, 'notebooks/census_boundaries/tl_2023_48_bg/tl_2023_48_bg.shp'),
    'bloomington': os.path.join(BASE_DIR, 'notebooks/census_boundaries/tl_2023_18_bg/tl_2023_18_bg.shp'),
    'cupertino': os.path.join(BASE_DIR, 'notebooks/census_boundaries/tl_2023_06_bg/tl_2023_06_bg.shp'),
    'surrey': os.path.join(BASE_DIR, 'notebooks/census_boundaries/lda_000b21a_e/lda_000b21a_e.shp'),
}

PLACE_BOUNDARY_PATHS = {
    'austin': {'path': os.path.join(BASE_DIR, 'notebooks/census_boundaries/tl_2023_48_place/tl_2023_48_place.shp'), 'col': 'NAME', 'val': 'Austin'},
    'bloomington': {'path': os.path.join(BASE_DIR, 'notebooks/census_boundaries/tl_2023_18_place/tl_2023_18_place.shp'), 'col': 'NAME', 'val': 'Bloomington'},
    'cupertino': {'path': os.path.join(BASE_DIR, 'notebooks/census_boundaries/tl_2023_06_place/tl_2023_06_place.shp'), 'col': 'NAME', 'val': 'Cupertino'},
    'surrey': {'path': os.path.join(BASE_DIR, 'notebooks/census_boundaries/lcsd000b21a_e/lcsd000b21a_e.shp'), 'col': 'CSDNAME', 'val': 'Surrey'},
}

def load_place_boundary(city):
    cfg = PLACE_BOUNDARY_PATHS.get(city)
    if not cfg:
        return None
    try:
        if not os.path.exists(cfg['path']):
            return None
        gdf = gpd.read_file(cfg['path'])
        if gdf.crs != 'EPSG:4326':
            gdf = gdf.to_crs('EPSG:4326')
        
        # Filter to city name
        subset = gdf[gdf[cfg['col']] == cfg['val']]
        if not subset.empty:
            geom = unary_union(subset.geometry)
            if not geom.is_valid:
                geom = make_valid(geom)
            return geom
    except Exception as e:
        print(f"Error loading {city} place boundary: {e}")
    return None

def draw_poly(ax, geom, facecolor, ec='#444444', lw=0.3):
    if geom is None or geom.is_empty:
        return
    if geom.geom_type == 'Polygon':
        if geom.exterior:
            xs, ys = geom.exterior.xy
            ax.add_patch(MplPolygon(list(zip(xs, ys)), facecolor=facecolor, edgecolor=ec, linewidth=lw, zorder=1))
    elif geom.geom_type == 'MultiPolygon':
        for p in geom.geoms:
            draw_poly(ax, p, facecolor, ec, lw)
    elif geom.geom_type == 'GeometryCollection':
        for part in geom.geoms:
            if part.geom_type in ('Polygon', 'MultiPolygon'):
                draw_poly(ax, part, facecolor, ec, lw)

def draw_boundary(ax, boundary):
    if boundary is None or boundary.is_empty:
        return
    try:
        if not boundary.is_valid:
            boundary = make_valid(boundary)
    except Exception:
        pass
        
    if boundary.geom_type == 'Polygon':
        if boundary.exterior:
            xs, ys = boundary.exterior.xy
            ax.plot(xs, ys, color='black', lw=1.2, zorder=10)
    elif boundary.geom_type == 'MultiPolygon':
        for p in boundary.geoms:
            if p and not p.is_empty:
                try:
                    if not p.is_valid:
                        p = make_valid(p)
                    if hasattr(p, 'exterior') and p.exterior:
                        xs, ys = p.exterior.xy
                        ax.plot(xs, ys, color='black', lw=1.2, zorder=10)
                except Exception:
                    pass

def add_scale_bar(ax, bounds, bar_km=2, city=None):
    bx0, by0, bx1, by1 = bounds
    mid_lat = (by0 + by1) / 2.0
    km_per_deg = 111.0 * np.cos(np.radians(mid_lat))
    bar_deg = bar_km / km_per_deg
    
    pad_x = (bx1 - bx0) * 0.06
    pad_y = (by1 - by0) * 0.05
    x0 = bx1 - pad_x - bar_deg
    y0 = by0 + pad_y
    
    if city == 'austin':
        y0 -= (by1 - by0) * 0.02
        
    ax.plot([x0, x0 + bar_deg], [y0, y0], color='black', lw=1.2, zorder=15)
    ax.text(x0 + bar_deg / 2.0, y0 + (by1 - by0) * 0.02, f'{bar_km} km', 
            ha='center', va='bottom', fontsize=15, color='black', zorder=15)

def get_area_km2(geom):
    """Geodesic area approximation in km2."""
    if geom is None or geom.is_empty:
        return 0
    # Consistent factor with generate_density_map_from_geojson.py
    return geom.area * 12391.0 * np.cos(np.radians(geom.centroid.y))

def main():
    cities = ['austin', 'bloomington', 'cupertino', 'surrey']
    
    print("Loading tree data...")
    # Read only requested cities directly to save memory. 
    # Use pandas chunking or we'll just read into memory if memory is sufficient. 
    df_trees = pd.read_csv(CSV_PATH, usecols=['city', 'long', 'lat'])
    df_trees = df_trees[df_trees['city'].isin(cities)].copy()
    
    # Need to keep it inside city boundary to match original exact trees if boundary exists, 
    # but spatial join generally accomplishes this to the BGs that are inside city.
    gdf_trees = gpd.GeoDataFrame(df_trees, geometry=gpd.points_from_xy(df_trees.long, df_trees.lat), crs="EPSG:4326")
    print(f"Loaded {len(gdf_trees)} trees across {cities}")

    city_results = {}
    
    for city in cities:
        print(f"Processing {city}...")
        trees_city = gdf_trees[gdf_trees['city'] == city]
        
        # Load Census block groups
        census_path = CENSUS_BOUNDARY_PATHS.get(city)
        if not census_path or not os.path.exists(census_path):
            print(f"Skipping {city}: Census shapefile missing.")
            continue
            
        bgs = gpd.read_file(census_path)
        if bgs.crs != 'EPSG:4326':
            bgs = bgs.to_crs('EPSG:4326')
            
        # Fix invalid geometries
        bgs['geometry'] = bgs['geometry'].apply(lambda g: make_valid(g) if (g and not g.is_valid) else g)
        bgs = bgs[bgs.geometry.notna() & ~bgs.geometry.is_empty]
        
        # Clip to place boundary to resemble exact map behavior
        place_geom = load_place_boundary(city)
        
        if place_geom and not place_geom.is_empty:
            trees_city = trees_city[trees_city.geometry.within(place_geom)]
            
            # For census block groups, clip to the official boundaries
            # In the original, city_boundary usually clipped the census_gdf if it was not empty.
            bgs = bgs[bgs.geometry.intersects(place_geom)]
        elif not trees_city.empty:
            # Fallback clip by tree bounds buffer
            bounds = trees_city.total_bounds
            tile_union = box(bounds[0], bounds[1], bounds[2], bounds[3]).buffer(0.003)
            bgs = bgs[bgs.geometry.intersects(tile_union)]
                
        # Load Coverage Mask if available
        coverage_mask = None
        meta_path = os.path.join(BASE_DIR, 'tile_metadata.csv')
        if os.path.exists(meta_path):
            try:
                df_meta = pd.read_csv(meta_path)
                keywords = {'austin':['austin'], 'bloomington':['bloomington'], 
                            'cupertino':['cupertino'], 'surrey':['surrey']}.get(city, [city])
                city_meta = df_meta[df_meta['filename'].str.lower().apply(lambda fn: any(kw in fn for kw in keywords))]
                if len(city_meta) > 0:
                    tile_boxes = [box(r.left, r.bottom, r.right, r.top) for r in city_meta.itertuples()]
                    coverage_mask = unary_union(tile_boxes).buffer(0.001)
            except Exception: pass

        # Calculate densities by clipping to boundary first
        if not bgs.empty:
            # Count trees per block group first (unclipped counts)
            joined = gpd.sjoin(trees_city, bgs, how="inner", predicate="intersects")
            joined = joined[~joined.index.duplicated(keep='first')]
            counts = joined.index_right.value_counts()
            
            bgs['trees'] = bgs.index.map(counts).fillna(0)
            
            # Now calculate clipped area and density
            def process_bg(row):
                geom = row.geometry
                if place_geom:
                    try:
                        geom = make_valid(geom).intersection(make_valid(place_geom))
                    except: pass
                
                area = get_area_km2(geom)
                return pd.Series([geom, area])

            bgs[['clipped_geometry', 'area_km2']] = bgs.apply(process_bg, axis=1)
            bgs['tree_density'] = bgs['trees'] / bgs['area_km2']
            bgs.loc[bgs['area_km2'] == 0, 'tree_density'] = 0
            bgs.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            total_trees = int(trees_city.shape[0])
        else:
            bgs['tree_density'] = 0
            bgs['trees'] = 0
            total_trees = 0

        city_results[city] = {
            'census_gdf': bgs,
            'boundary': place_geom,
            'coverage_mask': coverage_mask,
            'total_trees': total_trees,
        }
        print(f"  {city.title()} total trees: {total_trees}")
        gc.collect()
        
    print("Generating maps...")
    
    # Global settings
    vmax_dens = 1000
    vmin_dens = 0
    cmap_dens = plt.cm.YlGn
    norm_dens = _MplNorm(vmin=vmin_dens, vmax=vmax_dens)

    def get_color(dens):
        if pd.isna(dens):
            return '#E0E0E0'
        return cmap_dens(norm_dens(min(max(0, dens), vmax_dens)))

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 18,
    })

    CITY_LABELS = {
        'austin': 'Austin, TX', 'bloomington': 'Bloomington, IN',
        'cupertino': 'Cupertino, CA', 'surrey': 'Surrey, BC',
    }
    
    fig = plt.figure(figsize=(10.0, 8.0))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.05], hspace=0.30, wspace=0.10, figure=fig)
    


    # Plot
    for idx, (city, data) in enumerate(city_results.items()):
        row, col_idx = divmod(idx, 2)
        ax = fig.add_subplot(gs[row, col_idx])
        ax.set_facecolor('white')
        
        bgs = data['census_gdf']
        boundary = data['boundary']
        coverage_mask = data['coverage_mask']
        
        # Chloropleth
        if bgs is not None and len(bgs) > 0:
            for row_data in bgs.itertuples():
                if coverage_mask and not coverage_mask.intersects(row_data.geometry):
                    continue
                    
                geom = row_data.clipped_geometry
                if geom.is_empty: continue
                
                fc = get_color(row_data.tree_density)
                
                # Robustly handle types
                polys = []
                if geom.geom_type == 'Polygon':
                    polys = [geom]
                elif geom.geom_type == 'MultiPolygon':
                    polys = list(geom.geoms)
                elif geom.geom_type == 'GeometryCollection':
                    polys = [p for p in geom.geoms if p.geom_type in ['Polygon', 'MultiPolygon']]
                    flat = []
                    for p in polys:
                        if p.geom_type == 'MultiPolygon': flat.extend(list(p.geoms))
                        else: flat.append(p)
                    polys = flat

                for p in polys:
                    if p.is_empty: continue
                    ax.add_patch(MplPolygon(list(zip(*p.exterior.xy)), facecolor=fc, edgecolor='#444444', lw=0.3, zorder=1))
        
        # Boundary outline
        draw_boundary(ax, boundary)
        
        # Bounding box limits logic
        xmin, ymin, xmax, ymax = -180, -90, 180, 90
        if boundary and not boundary.is_empty:
            xmin, ymin, xmax, ymax = boundary.bounds
        elif bgs is not None and not bgs.empty:
            xmin, ymin, xmax, ymax = bgs.total_bounds
            
        if city == 'austin':
            centroid = boundary.centroid
            cx, cy = centroid.x, centroid.y
            span = 0.25
            ax.set_xlim(cx - span, cx + span)
            ax.set_ylim(cy - span, cy + span)
        elif city == 'surrey':
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
        
        BAR_CONFIGS = {'austin': 10, 'bloomington': 2, 'cupertino': 2, 'surrey': 5}
        bar_km = BAR_CONFIGS.get(city, 2)
        
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
            
        ax.set_title(CITY_LABELS.get(city, city.title()), fontsize=20, fontweight='bold', pad=6)
        
        subtitle = f'$n$ = {data["total_trees"]:,} trees'
        ax.text(0.5, -0.05, subtitle, transform=ax.transAxes, fontsize=16, color='#444444', style='italic', va='top', ha='center',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=1.0), zorder=15)
                
        # Pass the map bounds to add_scale_bar
        map_bounds = ax.get_xlim()[0], ax.get_ylim()[0], ax.get_xlim()[1], ax.get_ylim()[1]
        add_scale_bar(ax, map_bounds, bar_km=bar_km, city=city)

    # Logarithmic-style colorbar but linear color
    cbar_ax = fig.add_subplot(gs[:, 2])
    sm = plt.cm.ScalarMappable(cmap=cmap_dens, norm=norm_dens)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(r'Tree Density (trees km$^{-2}$)', fontsize=18, fontweight='bold')
    
    _ticks = [0, 200, 400, 600, 800, 1000]
    cbar.set_ticks(_ticks)
    _labels = [f'{int(t):,}' for t in _ticks]
    _labels[-1] = f'1,000+'
    cbar.set_ticklabels(_labels)
    cbar.ax.tick_params(labelsize=16)

    plt.tight_layout(rect=[0, 0, 1, 1])
    out_png = os.path.join(OUTPUT_DIR, 'csv_tree_density_map.png')
    out_pdf = os.path.join(OUTPUT_DIR, 'csv_tree_density_map.pdf')
    fig.savefig(out_png, dpi=300, facecolor='white', bbox_inches='tight', pad_inches=0.05)
    fig.savefig(out_pdf, facecolor='white', bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"Maps saved to:\n  {out_png}\n  {out_pdf}")

if __name__ == "__main__":
    main()
