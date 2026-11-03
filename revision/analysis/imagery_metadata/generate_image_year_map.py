import os
import glob
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, box
from shapely.validation import make_valid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
from scipy.interpolate import make_interp_spline

# =============================================================================
# Configuration & Dynamic Directory Resolution
# =============================================================================
BASE_DIR = os.environ.get("TREE_MAPPING_BASE_DIR", os.path.dirname(os.path.abspath(__file__)))

SEARCH_BASES = [
    BASE_DIR,
    os.path.join(BASE_DIR, "sherlock_venv"),
    os.path.join(BASE_DIR, "notebooks"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sherlock_venv"),
    "/scratch/groups/cfield/myenv_scratch/notebooks_scratch"
]

def find_file(rel_path):
    for base in SEARCH_BASES:
        p = os.path.join(base, rel_path)
        if os.path.exists(p):
            return p
    return None

def find_city_imagery_dir(city):
    candidates = [

        os.path.join(BASE_DIR, "Training_06062024", city.lower()),
        os.path.join(BASE_DIR, "cutouts", city.lower()),
        f"/scratch/groups/cfield/myenv_scratch/notebooks_scratch/Training_06062024/{city.lower()}/",
        f"/scratch/groups/cfield/myenv_scratch/notebooks_scratch/cutouts/{city.lower()}/"
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

CITY_CONFIGS = {
    'Austin': {
        'shp_rel': 'notebooks/census_boundaries/tl_2023_48_place/tl_2023_48_place.shp',
        'filter_col': 'NAME',
        'filter_val': 'Austin',
        'state_desc': 'Austin, TX (Humid subtropical, Cfa)'
    },
    'Bloomington': {
        'shp_rel': 'notebooks/census_boundaries/tl_2023_18_place/tl_2023_18_place.shp',
        'filter_col': 'NAME',
        'filter_val': 'Bloomington',
        'state_desc': 'Bloomington, IN (Humid continental, Dfa)'
    },
    'Cupertino': {
        'shp_rel': 'notebooks/census_boundaries/tl_2023_06_place/tl_2023_06_place.shp',
        'filter_col': 'NAME',
        'filter_val': 'Cupertino',
        'state_desc': 'Cupertino, CA (Mediterranean, Csb)'
    },
    'Surrey': {
        'shp_rel': 'notebooks/census_boundaries/lcsd000b21a_e/lcsd000b21a_e.shp',
        'filter_col': 'CSDNAME',
        'filter_val': 'Surrey',
        'state_desc': 'Surrey, BC (Temperate oceanic, Cfb)'
    }
}

YEAR_BINS = []
BIN_LABELS = []

def set_dynamic_bins(years, num_bins=4):
    global YEAR_BINS, BIN_LABELS
    if len(years) == 0: return
    min_yr = max(2000, int(min(years)))
    max_yr = 2024
    total_range = max_yr - min_yr + 1
    step = total_range / num_bins
    YEAR_BINS = []
    BIN_LABELS = []
    current = min_yr
    for i in range(num_bins):
        start = int(current)
        end = max_yr + 1 if i == num_bins - 1 else int(current + step)
        if end <= start: end = start + 1
        YEAR_BINS.append((start, end))
        BIN_LABELS.append(f"{start}–{end-1}" if end - 1 > start else f"{start}")
        current = end

def get_year_bin(year):
    for i, (start, end) in enumerate(YEAR_BINS):
        if start <= year < end:
            return i
    return None

# =============================================================================
# Day of Year (DOY) Phenological Bins & Sensor Configurations
# =============================================================================
# DOY bins covering the full calendar range including November acquisitions
DOY_BINS = [0, 120, 160, 190, 220, 366]
DOY_LABELS = ['44–120', '121–160', '161–190', '191–220', '221–292']
DOY_COLORS = ['#F0E5F4', '#D5B4DF', '#B77AC8', '#88499F', '#54236F']

def get_doy_bin(doy):
    try:
        d = float(doy)
        for i, threshold in enumerate(DOY_BINS[1:]):
            if d <= threshold:
                return i
        return len(DOY_BINS) - 2
    except:
        return 0

SENSOR_NAMES = {
    'GE01': 'GeoEye-1',
    'WV02': 'WorldView-2',
    'WV03': 'WorldView-3',
    'QB02': 'QuickBird-2'
}

SENSOR_COLORS = {
    'GeoEye-1': '#009E73',
    'QuickBird-2': '#E69F00',
    'WorldView-2': '#56B4E9',
    'WorldView-3': '#D55E00'
}

OFF_NADIR_BINS = [0, 5, 15, 25, 90]
OFF_NADIR_LABELS = ['< 5°', '5–15°', '15–25°', '> 25°']
OFF_NADIR_COLORS = ['#FFF0BE', '#F5C66B', '#D98B45', '#A94F35']

CITY_SCALE_BARS = {
    'Austin': 10,
    'Bloomington': 5,
    'Surrey': 5,
    'Cupertino': 2
}

# =============================================================================
# Data Loading & Ingestion
# =============================================================================
def load_city_gpkgs(city):
    """Recursively loads mosaic cutline GPKGs for a city."""
    base_dir = find_city_imagery_dir(city)
    if not base_dir or not os.path.exists(base_dir):
        return gpd.GeoDataFrame()
    gpkg_files = glob.glob(os.path.join(base_dir, "**/*_mosaic_cutlines.gpkg"), recursive=True)
    if not gpkg_files:
        return gpd.GeoDataFrame()
    gdfs = []
    for f in gpkg_files:
        try:
            gdf = gpd.read_file(f, engine='pyogrio')
            if not gdf.empty:
                gdf['city_assigned'] = city
                gdfs.append(gdf)
        except:
            pass
    if not gdfs:
        return gpd.GeoDataFrame()
    combined = pd.concat(gdfs, ignore_index=True)
    return gpd.GeoDataFrame(combined, crs=gdfs[0].crs)

def extract_data_from_gpkgs():
    all_city_gdfs = []
    for city in CITY_CONFIGS.keys():
        city_gdf = load_city_gpkgs(city)
        if not city_gdf.empty:
            all_city_gdfs.append(city_gdf)
    if not all_city_gdfs:
        return gpd.GeoDataFrame()
    gdf = pd.concat(all_city_gdfs, ignore_index=True)
    gdf = gpd.GeoDataFrame(gdf, crs=all_city_gdfs[0].crs)
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    
    if 'CAT_ID' in gdf.columns:
        gdf['catalog_id'] = gdf['CAT_ID']
    def parse_year(val):
        try: return int(str(val)[:4])
        except: return None
    gdf['year'] = gdf['ACQDATE'].apply(parse_year)
    
    if 'SENSOR' in gdf.columns:
        gdf['sensor'] = gdf['SENSOR']
    gdf['sensor_full'] = gdf['sensor'].map(SENSOR_NAMES).fillna(gdf['sensor'])
    
    def get_off_nadir_bin(val):
        try:
            v = float(val)
            for i, threshold in enumerate(OFF_NADIR_BINS[1:]):
                if v <= threshold: return i
            return len(OFF_NADIR_BINS) - 2
        except: return 0
    gdf['off_nadir_bin'] = gdf['OFF_NADIR'].apply(get_off_nadir_bin) if 'OFF_NADIR' in gdf.columns else 0
    
    if 'ACQDATE' in gdf.columns:
        gdf['acq_dt'] = pd.to_datetime(gdf['ACQDATE'], errors='coerce')
        gdf['doy'] = gdf['acq_dt'].dt.dayofyear
        gdf['doy_bin'] = gdf['doy'].apply(get_doy_bin)
        gdf = gdf.sort_values('acq_dt', ascending=True)
    return gdf

def load_ndvi_curves():
    """Loads empirical QA-filtered 16-day MODIS NDVI curves from CSV."""
    candidates = [
        "city_seasonal_ndvi_curves.csv",
        os.path.join(BASE_DIR, "city_seasonal_ndvi_curves.csv"),

        os.path.join(BASE_DIR, "analysis_output", "city_seasonal_ndvi_curves.csv")
    ]
    for p in candidates:
        if os.path.exists(p):
            return pd.read_csv(p)
    return None

def build_ndvi_spline(city_df):
    clean = city_df.sort_values('doy').drop_duplicates('doy')
    doy_raw = list(clean['doy'].values)
    ndvi_raw = list(clean['ndvi'].values)
    if doy_raw[-1] < 365:
        doy_raw.append(365); ndvi_raw.append(ndvi_raw[0])
    if doy_raw[0] > 1:
        doy_raw.insert(0, 1); ndvi_raw.insert(0, ndvi_raw[-1])
    doy_grid = np.linspace(1, 365, 365)
    spl = make_interp_spline(doy_raw, ndvi_raw, k=3)
    ndvi_grid = spl(doy_grid)
    return doy_grid, ndvi_grid, spl

# =============================================================================
# Standalone Phenology Summary Figure (Supplementary Figure S8)
# =============================================================================
PHENOLOGY_YEAR_COLORS = {
    2009: '#1f78b4',  # Deep Blue
    2013: '#00a6ca',  # Bright Cyan / Cerulean
    2015: '#e6ab02',  # Amber Gold
    2016: '#ff7f00',  # Vibrant Orange
    2017: '#e31a1c',  # Crimson Red
    2019: '#8c564b',  # Warm Brown
    2021: '#e7298a',  # Deep Rose
    2023: '#333333'   # Dark Charcoal / Slate
}

def generate_phenology_summary_figure(footprints_gdf, ndvi_df, cities_gdf):
    """
    Generates Supplementary Figure S8: a dedicated, publication-quality 2x2 multi-panel
    figure communicating image acquisition timing relative to seasonal vegetation phenology.
    Features:
      - Large, prominent, publication-ready typography across all elements.
      - One shared, bold "Normalized Difference Vegetation Index (NDVI)" title vertically centered
        along the left side of the entire figure (14.0 pt).
      - One shared, bold "Month" title centered horizontally beneath the bottom row of panels (14.0 pt).
      - Bold city headings with Koppen climate classifications (14.0 pt).
      - Numerical y-axis ticks (11.5 pt) and January-December x-axis ticks (11.0 pt) preserved on all four panels.
      - Distinct, visually distinguishable acquisition-year marker colors mapped across all cities.
      - Circular marker area proportional to municipal coverage percentage.
      - Green empirical MODIS NDVI spline curves with purple dotted peak greenness lines.
      - Split bicolor markers for co-occurring acquisition dates (e.g., Austin DOY 225).
      - Three compact, frameless legends centered below the figure (11.8 - 12.2 pt text and bold titles).
      - High-resolution 300 DPI PNG and vector PDF exports.
    """
    if ndvi_df is None:
        return None
        
    FIG_WIDTH = 11.2
    FIG_HEIGHT = 9.6  # Generous publication height accommodating larger typography comfortably
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=300)
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica'],
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.0,
        'text.color': 'black',
        'axes.labelcolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black'
    })

    # outer gridspec with ample margins and comfortable inter-row spacing
    outer_gs = fig.add_gridspec(2, 2, left=0.095, right=0.965, top=0.955, bottom=0.21, wspace=0.22, hspace=0.32)
    
    city_list = list(CITY_CONFIGS.keys())
    city_panel_configs = {
        'Austin': {'y_lim': (0.25, 0.75), 'y_ticks': [0.3, 0.4, 0.5, 0.6, 0.7]},
        'Bloomington': {'y_lim': (0.15, 0.65), 'y_ticks': [0.2, 0.3, 0.4, 0.5, 0.6]},
        'Cupertino': {'y_lim': (0.22, 0.52), 'y_ticks': [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]},
        'Surrey': {'y_lim': (0.35, 0.75), 'y_ticks': [0.4, 0.5, 0.6, 0.7]}
    }

    scale_factor = 3.5
    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    acq_summary_rows = []
    
    ndvi_axes = []
    strip_axes = []

    for idx, city in enumerate(city_list):
        r, c = idx // 2, idx % 2
        cell_gs = outer_gs[r, c].subgridspec(2, 1, height_ratios=[4.2, 0.8], hspace=0.05)
        
        ax_ndvi = fig.add_subplot(cell_gs[0])
        ax_strip = fig.add_subplot(cell_gs[1], sharex=ax_ndvi)
        ndvi_axes.append(ax_ndvi)
        strip_axes.append(ax_strip)
        
        cinfo = CITY_CONFIGS[city]
        pcfg = city_panel_configs[city]
        city_geom = make_valid(cities_gdf[city].geometry.iloc[0])
        city_area = city_geom.area
        
        # 1. Empirical MODIS NDVI curve
        city_ndvi = ndvi_df[ndvi_df['city'] == city]
        doy_grid, ndvi_grid, spl = build_ndvi_spline(city_ndvi)
        peak_doy = int(doy_grid[np.argmax(ndvi_grid)])

        # Green empirical MODIS NDVI curve
        ax_ndvi.plot(doy_grid, ndvi_grid, color='#238b45', linewidth=2.6, zorder=3)
        
        # Purple dotted line for peak annual NDVI
        ax_ndvi.axvline(peak_doy, color='#762a83', linestyle=':', linewidth=1.6, zorder=2)
        ax_strip.axvline(peak_doy, color='#762a83', linestyle=':', linewidth=1.4, alpha=0.45, zorder=1)

        # 2. Extract verified municipal acquisitions
        city_foots = footprints_gdf[footprints_gdf['city_assigned'] == city].copy()
        city_acqs = []
        if not city_foots.empty:
            city_foots['date_str'] = city_foots['acq_dt'].dt.strftime('%Y-%m-%d')
            city_foots['geometry'] = city_foots.geometry.apply(make_valid)
            city_foots = city_foots[city_foots.intersects(city_geom)].copy()
            clipped_geoms = []
            for geom in city_foots.geometry:
                try:
                    clipped_geoms.append(geom.intersection(city_geom))
                except:
                    clipped_geoms.append(geom.buffer(0).intersection(city_geom))
            city_foots['geometry'] = clipped_geoms
            city_foots = city_foots[~city_foots.geometry.is_empty]
            
            for (dt_str, sens, yr, adoy), grp in city_foots.groupby(['date_str', 'sensor_full', 'year', 'doy']):
                d_union = grp.geometry.union_all() if hasattr(grp.geometry, 'union_all') else grp.geometry.unary_union
                d_union = make_valid(d_union)
                pct = (d_union.area / city_area) * 100
                if pct > 0.05:
                    city_acqs.append({
                        'date': dt_str,
                        'doy': int(adoy),
                        'sensor': sens,
                        'pct': pct,
                        'year': int(yr)
                    })
                    acq_summary_rows.append({
                        'City': city,
                        'Acquisition Date': dt_str,
                        'DOY': int(adoy),
                        'Year': int(yr),
                        'Sensor': sens,
                        'Municipal Area (%)': round(pct, 2)
                    })

        doy_groups = {}
        for acq in city_acqs:
            doy_groups.setdefault(acq['doy'], []).append(acq)

        # Markers on NDVI curve with acquisition-year colors
        for adoy, acqs in doy_groups.items():
            aval = float(spl(adoy))
            if len(acqs) == 1:
                acq = acqs[0]
                col = PHENOLOGY_YEAR_COLORS.get(acq['year'], '#333333')
                ax_ndvi.scatter([adoy], [aval], color=col, s=56, edgecolor='black', linewidth=0.7, zorder=6)
            elif len(acqs) == 2:
                # Two acquisitions on same DOY (Austin DOY 225: 2017 & 2019)
                col1 = PHENOLOGY_YEAR_COLORS.get(acqs[0]['year'], '#333333')
                col2 = PHENOLOGY_YEAR_COLORS.get(acqs[1]['year'], '#333333')
                ax_ndvi.scatter([adoy], [aval], marker=matplotlib.markers.MarkerStyle('o', fillstyle='left'),
                                color=col1, s=60, edgecolor='black', linewidth=0.7, zorder=6)
                ax_ndvi.scatter([adoy], [aval], marker=matplotlib.markers.MarkerStyle('o', fillstyle='right'),
                                color=col2, s=60, edgecolor='black', linewidth=0.7, zorder=6)
            else:
                for acq in acqs:
                    col = PHENOLOGY_YEAR_COLORS.get(acq['year'], '#333333')
                    ax_ndvi.scatter([adoy], [aval], color=col, s=56, edgecolor='black', linewidth=0.7, zorder=6)

        # 3. Compact Timeline Strip
        ax_strip.axhline(0, color='#cccccc', linewidth=0.9, linestyle='-', zorder=1)
        ax_strip.set_ylim(-0.48, 0.48)
        ax_strip.set_yticks([])
        ax_strip.set_facecolor('#ffffff')
        
        for adoy, acqs in doy_groups.items():
            if len(acqs) == 1:
                acq = acqs[0]
                sz = max(14, scale_factor * acq['pct'])
                col = PHENOLOGY_YEAR_COLORS.get(acq['year'], '#333333')
                ax_strip.scatter([adoy], [0], color=col, s=sz, edgecolor='black', linewidth=0.6, alpha=0.9, zorder=4)
            else:
                offsets = [-0.18, 0.18] if len(acqs) == 2 else np.linspace(-0.25, 0.25, len(acqs))
                ax_strip.plot([adoy, adoy], [offsets[0], offsets[-1]], color='#999999', linestyle=':', linewidth=0.8, zorder=2)
                for i_sub, acq in enumerate(acqs):
                    sz = max(14, scale_factor * acq['pct'])
                    y_off = offsets[i_sub]
                    col = PHENOLOGY_YEAR_COLORS.get(acq['year'], '#333333')
                    ax_strip.scatter([adoy], [y_off], color=col, s=sz, edgecolor='black', linewidth=0.6, alpha=0.9, zorder=4)

        # Panel Headings: bold, prominent, exact Koppen classification (14.0 pt)
        ax_ndvi.set_title(cinfo['state_desc'], loc='left', fontsize=14.0, fontweight='bold', color='black', pad=9)
        ax_ndvi.set_xlim(1, 365)
        ax_ndvi.set_ylim(pcfg['y_lim'])
        ax_ndvi.set_yticks(pcfg['y_ticks'])
        
        # Keep numerical tick labels on all four panels (11.5 pt)
        ax_ndvi.tick_params(axis='y', labelsize=11.5, colors='black', left=True, labelleft=True)
        ax_ndvi.grid(True, linestyle=':', alpha=0.45, color='#b0b0b0', zorder=0)
        plt.setp(ax_ndvi.get_xticklabels(), visible=False)
        ax_ndvi.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
        
        # Individual panel y-axis titles are cleared (replaced by single shared title on left side)
        ax_ndvi.set_ylabel('')

        # Formatting Timeline strip: keep January-December tick labels on all four panels (11.0 pt)
        ax_strip.set_xlim(1, 365)
        ax_strip.set_xticks(month_starts)
        ax_strip.set_xticklabels(month_names, fontsize=11.0, color='black', fontweight='medium')
        ax_strip.tick_params(axis='x', which='both', bottom=True, top=False, labelsize=11.0, colors='black', pad=4)
        ax_strip.grid(True, axis='x', linestyle=':', alpha=0.35, color='#c0c0c0', zorder=0)
        ax_strip.set_xlabel('')

    fig.canvas.draw()
    
    pos_top_left_ndvi = ndvi_axes[0].get_position()
    pos_bot_left_ndvi = ndvi_axes[2].get_position()
    pos_bot_left_strip = strip_axes[2].get_position()
    pos_bot_right_strip = strip_axes[3].get_position()

    # 1. Single Shared Y-axis Title: (14.0 pt bold)
    # Vertically centered along the left side of the entire figure relative to the four main NDVI panels and strips
    y_center_left = (pos_top_left_ndvi.y1 + pos_bot_left_strip.y0) / 2.0
    fig.text(0.026, y_center_left, 'Normalized Difference Vegetation Index (NDVI)',
             va='center', ha='center', rotation='vertical',
             fontsize=14.0, fontweight='bold', color='black')

    # 2. Single Shared X-axis Title: (14.0 pt bold)
    # Centered horizontally beneath the bottom row of panels and above the legends
    x_center_bottom = (pos_bot_left_strip.x0 + pos_bot_right_strip.x1) / 2.0
    y_month = pos_bot_left_strip.y0 - 0.046
    fig.text(x_center_bottom, y_month, 'Month', ha='center', va='top',
             fontsize=14.0, fontweight='bold', color='black')

    # Three clean, compact, frameless single-row legends below the figure
    # Row 1: Phenology Reference Lines (12.2 pt)
    leg_handles_ref = [
        Line2D([0], [0], color='#238b45', linewidth=3.0, label='Seasonal NDVI (MODIS)'),
        Line2D([0], [0], color='#762a83', linestyle=':', linewidth=2.2, label='Peak annual NDVI')
    ]
    
    # Row 2: Acquisition Year Legend (8 distinct colors across all cities, 11.8 pt)
    unique_years = sorted(list(PHENOLOGY_YEAR_COLORS.keys()))
    leg_handles_years = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=PHENOLOGY_YEAR_COLORS[yr], markeredgecolor='black',
               markeredgewidth=0.6, markersize=9.0, label=str(yr))
        for yr in unique_years
    ]
    
    # Row 3: Acquisition Coverage (Bubble Sizes, 11.8 pt)
    size_pcts = [10, 25, 50, 100]
    leg_handles_cov = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#888888', markeredgecolor='black',
               markeredgewidth=0.6, markersize=np.sqrt(scale_factor * p), label=f'{p}%')
        for p in size_pcts
    ]

    # Legend 1: Reference curves
    leg_ref = fig.legend(handles=leg_handles_ref, loc='center', bbox_to_anchor=(x_center_bottom, 0.124),
                         ncol=2, frameon=False, fontsize=12.2, handlelength=2.5, columnspacing=2.6)
    
    # Legend 2: Acquisition Years
    leg_yr = fig.legend(handles=leg_handles_years, loc='center', bbox_to_anchor=(x_center_bottom + 0.065, 0.073),
                        ncol=8, frameon=False, fontsize=11.8, handlelength=1.1, handletextpad=0.35, columnspacing=1.3)
    
    # Legend 3: Municipal Coverage
    leg_cov = fig.legend(handles=leg_handles_cov, loc='center', bbox_to_anchor=(x_center_bottom + 0.10, 0.025),
                         ncol=4, frameon=False, fontsize=11.8, handlelength=1.5, handletextpad=0.6, columnspacing=2.6)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv_fig = fig.transFigure.inverted()
    
    bb_yr = inv_fig.transform_bbox(leg_yr.get_window_extent(renderer))
    bb_cov = inv_fig.transform_bbox(leg_cov.get_window_extent(renderer))
    
    # Draw bold inline titles with perfect alignment (12.2 pt bold)
    fig.text(bb_yr.x0 - 0.012, (bb_yr.y0 + bb_yr.y1) / 2.0, 'Acquisition year:',
             ha='right', va='center', fontsize=12.2, fontweight='bold', color='black')
    
    fig.text(bb_cov.x0 - 0.012, (bb_cov.y0 + bb_cov.y1) / 2.0, 'Acquisition coverage:',
             ha='right', va='center', fontsize=12.2, fontweight='bold', color='black')

    out_png = "image_acquisition_phenology_summary.png"
    out_pdf = "image_acquisition_phenology_summary.pdf"
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    print(f"Saved publication phenology summary figure to: {out_png} and {out_pdf}")
    plt.close()
    
    return pd.DataFrame(acq_summary_rows)

def main():
    print("Aggregating mosaic cutlines from GPKG files...")
    footprints_gdf = extract_data_from_gpkgs()
    if footprints_gdf.empty:
        print("ERROR: No footprints found.")
        return

    set_dynamic_bins(footprints_gdf['year'].dropna().unique(), num_bins=4)
    footprints_gdf['year_bin'] = footprints_gdf['year'].apply(get_year_bin)
    
    ndvi_df = load_ndvi_curves()
    if ndvi_df is not None:
        print(f"Loaded seasonal NDVI curves for: {list(ndvi_df['city'].unique())}")
    
    cities_gdf = {}
    city_bounds = {}
    for city, cfg in CITY_CONFIGS.items():
        shp_path = find_file(cfg['shp_rel'])
        if shp_path and os.path.exists(shp_path):
            gdf = gpd.read_file(shp_path)
            gdf = gdf[gdf[cfg['filter_col']] == cfg['filter_val']].copy()
            if gdf.crs != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")
            gdf['geometry'] = gdf.geometry.buffer(0)
            cities_gdf[city] = gdf
            city_bounds[city] = gdf.total_bounds

    city_view_extents = {}
    for city, bounds in city_bounds.items():
        minx, miny, maxx, maxy = bounds
        dx = maxx - minx
        dy = maxy - miny
        pad = 1.02 if city == 'Cupertino' else 1.10
        city_view_extents[city] = {
            'win_size_x': dx * pad,
            'win_size_y': dy * pad,
            'cx': (minx + maxx) / 2,
            'cy': (miny + maxy) / 2
        }

    # 4 Rows, 4 Columns (Austin, Bloomington, Cupertino, Surrey):
    # [Row 0: Year, Row 1: DOY, Row 2: Sensor, Row 3: Off-Nadir]
    # No insets on map panels as requested
    FIG_WIDTH = 7.08661 # 180mm
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_WIDTH * 1.25))
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica']})
    
    gs = fig.add_gridspec(4, 6, width_ratios=[0.05, 1, 1, 1, 1, 0.48], wspace=0.08, hspace=0.09)
    
    # Determine verified minimum and maximum acquisition DOYs from mapped municipal cutlines
    mapped_doys = []
    for city in CITY_CONFIGS:
        if city in cities_gdf:
            city_gdf = cities_gdf[city]
            city_boundary = city_gdf.geometry.buffer(0).union_all() if hasattr(city_gdf.geometry, 'union_all') else city_gdf.geometry.buffer(0).unary_union
            city_foots = footprints_gdf[footprints_gdf['city_assigned'] == city].copy()
            city_foots = city_foots[city_foots.intersects(city_boundary)].copy()
            if not city_foots.empty and 'doy' in city_foots.columns:
                mapped_doys.extend(city_foots['doy'].dropna().astype(int).tolist())
    
    if mapped_doys:
        min_doy = min(mapped_doys)
        max_doy = max(mapped_doys)
        doy_legend_labels = [f"{min_doy}–120", '121–160', '161–190', '191–220', f"221–{max_doy}"]
        print(f"Verified mapped acquisition DOY range: {min_doy} to {max_doy} (min: DOY {min_doy}, max: DOY {max_doy})")
    else:
        doy_legend_labels = DOY_LABELS

    characteristics = [
        {'id': 'year', 'label': 'Acquisition Year', 'cmap': ListedColormap(['#deebf7', '#9ecae1', '#4292c6', '#08519c']), 'bins': BIN_LABELS},
        {'id': 'doy', 'label': 'Acquisition Day of Year (DOY)', 'cmap': ListedColormap(DOY_COLORS), 'bins': doy_legend_labels},
        {'id': 'sensor', 'label': 'Dominant Sensor', 'cmap': ListedColormap([SENSOR_COLORS[s] for s in sorted(list(SENSOR_NAMES.values()))]), 'bins': sorted(list(SENSOR_NAMES.values()))},
        {'id': 'off_nadir', 'label': 'Off-Nadir Angle', 'cmap': ListedColormap(OFF_NADIR_COLORS), 'bins': OFF_NADIR_LABELS}
    ]
    
    city_list = list(CITY_CONFIGS.keys())
    panel_idx = 0
    alphabet = 'abcdefghijklmnopqrstuvwxyz'

    for row_idx, char in enumerate(characteristics):
        ax_label = fig.add_subplot(gs[row_idx, 0])
        ax_label.axis('off')
        ax_label.text(0.5, 0.5, char['label'], rotation=90, va='center', ha='center', fontsize=7.5, fontweight='bold')
        
        n_colors = len(char['bins'])
        norm = BoundaryNorm(range(n_colors + 1), n_colors)
        
        for col_idx, city in enumerate(city_list):
            ax = fig.add_subplot(gs[row_idx, col_idx + 1])
            city_gdf = cities_gdf[city]
            city_boundary = city_gdf.geometry.buffer(0).union_all() if hasattr(city_gdf.geometry, 'union_all') else city_gdf.geometry.buffer(0).unary_union
            city_boundary = make_valid(city_boundary)
            
            extent = city_view_extents[city]
            win_x, win_y = extent['win_size_x'], extent['win_size_y']
            cx, cy = extent['cx'], extent['cy']
            ax.set_xlim(cx - win_x/2, cx + win_x/2)
            ax.set_ylim(cy - win_y/2, cy + win_y/2)
            ax.set_aspect(1.0/np.cos(np.radians(cy)), adjustable='datalim')
            ax.set_anchor('C')
            ax.set_xticks([]); ax.set_yticks([])
            
            ax.set_facecolor('#ffffff')
            gpd.GeoSeries([city_boundary]).plot(ax=ax, color='#d9d9d9', zorder=1)
            
            city_foots = footprints_gdf[footprints_gdf['city_assigned'] == city].copy()
            if not city_foots.empty:
                city_foots['geometry'] = city_foots.geometry.apply(make_valid)
                city_foots = city_foots[city_foots.intersects(city_boundary)].copy()
                valid_geoms = []
                for geom in city_foots.geometry:
                    try:
                        valid_geoms.append(geom.intersection(city_boundary))
                    except:
                        valid_geoms.append(geom.buffer(0).intersection(city_boundary))
                city_foots['geometry'] = valid_geoms
                city_foots = city_foots[~city_foots.geometry.is_empty]
                
                if char['id'] == 'year':
                    city_foots.plot(ax=ax, column='year_bin', cmap=char['cmap'], norm=norm, linewidth=0.1, edgecolor='black', zorder=2)
                elif char['id'] == 'doy':
                    # DOY map without insets (clean publication view)
                    city_foots.plot(ax=ax, column='doy_bin', cmap=char['cmap'], norm=norm, linewidth=0.1, edgecolor='black', zorder=2)
                elif char['id'] == 'sensor':
                    s_map = {s: i for i, s in enumerate(char['bins'])}
                    city_foots['sensor_idx'] = city_foots['sensor_full'].map(s_map).fillna(0).astype(int)
                    city_foots.plot(ax=ax, column='sensor_idx', cmap=char['cmap'], norm=norm, linewidth=0.1, edgecolor='black', zorder=2)
                elif char['id'] == 'off_nadir':
                    city_foots.plot(ax=ax, column='off_nadir_bin', cmap=char['cmap'], norm=norm, linewidth=0.1, edgecolor='black', zorder=2)
            
            gpd.GeoSeries([city_boundary]).plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.5, zorder=3)
            if row_idx == 0:
                ax.set_title(city, fontsize=7.5, pad=6)
            
            ax.text(0.04, 0.96, f"{alphabet[panel_idx]}", transform=ax.transAxes, 
                    va='top', ha='left', fontsize=9, fontweight='bold', zorder=10,
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=1))
            panel_idx += 1
            
            # Scale bar
            clat = cy
            kpl_lon = 111.32 * np.cos(np.radians(clat))
            b_km = CITY_SCALE_BARS.get(city, 5)
            b_deg = b_km / kpl_lon
            
            bx, _ = ax.transData.inverted().transform(ax.transAxes.transform((0.05, 0.05)))
            transform = ax.get_xaxis_transform()
            ax.plot([bx, bx + b_deg], [0.05, 0.05], transform=transform, color='black', linewidth=0.6, zorder=5)
            ax.text(bx + b_deg/2, 0.08, f'{b_km} km', transform=transform,
                    ha='center', va='bottom', fontsize=5, zorder=5)

        # Legend
        ax_leg = fig.add_subplot(gs[row_idx, 5])
        ax_leg.axis('off')
        patches = [mpatches.Patch(facecolor=char['cmap'].colors[i], label=char['bins'][i], linewidth=0.2, edgecolor='black') for i in range(len(char['bins']))]
        ax_leg.legend(handles=patches, loc='center left', bbox_to_anchor=(0.0, 0.5), frameon=False, fontsize=6.0, labelspacing=0.7)

    out_file = "image_characteristics_map_final.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.savefig("image_year_map.png", dpi=300, bbox_inches='tight')
    plt.savefig("image_year_characteristics.png", dpi=300, bbox_inches='tight')
    plt.savefig("image_characteristics_map_final.pdf", bbox_inches='tight')
    print(f"Saved finalized 4-row characteristics map to:\n  - {out_file}\n  - image_year_map.png\n  - image_characteristics_map_final.pdf")
    plt.close()

    # Generate standalone phenology summary figure & print verification table
    table_df = generate_phenology_summary_figure(footprints_gdf, ndvi_df, cities_gdf)
    if table_df is not None and not table_df.empty:
        print("\n" + "="*80)
        print("VERIFIED ACQUISITION METADATA TABLE (NON-OVERLAPPING MUNICIPAL COVERAGE)")
        print("="*80)
        print(table_df.to_string(index=False))
        print("="*80 + "\n")
        table_df.to_csv("verified_municipal_acquisitions_table.csv", index=False)
        table_df.to_csv(os.path.join(BASE_DIR, "verified_municipal_acquisitions_table.csv"), index=False)

if __name__ == "__main__":
    main()
