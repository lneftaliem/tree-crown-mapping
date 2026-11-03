#!/usr/bin/env python3
"""
generate_redlining_figures.py

Generates publication-ready scientific figures and audited CSV tables adhering to
Nature Portfolio standards for the exploratory Austin HOLC redlining and urban
tree canopy analysis:
- Figure 1: Historical HOLC grades, contemporary tree distribution, and municipal context (PNG + PDF)
- Figure 2: Individual-polygon tree metrics across historical HOLC grades (PNG + PDF)
- Figure 3: Polygon 9093 influence and leave-one-polygon-out sensitivity analysis (PNG + PDF)
- Audited CSV tables: polygon-level metrics, grade-level aggregates, and sensitivity analysis.
"""

import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, Patch
import matplotlib.lines as mlines

# Nature Portfolio Typography and Aesthetic Settings (Enlarged and Bolder)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica Neue', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.size': 10.5,
    'axes.labelsize': 12.0,
    'axes.labelweight': 'bold',
    'axes.titlesize': 13.5,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 10.5,
    'ytick.labelsize': 10.5,
    'legend.fontsize': 10.0,
    'legend.title_fontsize': 10.5,
    'figure.titlesize': 14.0,
    'axes.linewidth': 0.9,
    'xtick.major.width': 0.9,
    'ytick.major.width': 0.9,
    'xtick.major.size': 4.0,
    'ytick.major.size': 4.0,
    'lines.linewidth': 1.6
})

# Inputs are committed to this repository and staged by repro_data:
#   data/redlining/austin_redlining_data.json   (Mapping Inequality HOLC polygons)
#   data/boundaries/austin_exact_*.geojson
#   data/tree_points/austin_ripley_points_with_income.geojson.gz.part*
# Outputs go to $TREE_MAPPING_BASE_DIR/analysis_output/redlining_figures/.
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
import repro_data  # noqa: E402

BASE_DIR = repro_data.BASE_DIR
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_output", "redlining_figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

HOLC_PATH = os.path.join(repro_data.REDLINING_STAGE_DIR, "austin_redlining_data.json")
EXACT_BG_PATH = os.path.join(repro_data.BOUNDARIES_STAGE_DIR, "austin_exact_block_groups.geojson")
EXACT_BND_PATH = os.path.join(repro_data.BOUNDARIES_STAGE_DIR, "austin_exact_boundary.geojson")

# Panel (b) of Figure 1 shows trees in a fixed window around the historical
# core: 0.142 deg of latitude tall, centred on lon -97.745 and the HOLC
# latitude midpoint, with the panel's 7.00 x 5.32 inch aspect ratio.
PANEL_B_SPAN_LAT = 0.142
PANEL_B_ASPECT = 7.00 / (0.76 * 7.0)
PANEL_B_CENTER_LON = -97.745

# Canonical, publication-calibrated HOLC colors
HOLC_COLORS = {
    'A': '#76a865',  # Olive Green / Grade A ("Best")
    'B': '#7cb5bd',  # Muted Slate Blue / Grade B ("Still Desirable")
    'C': '#d4ac0d',  # Warm Ochre Yellow / Grade C ("Definitely Declining")
    'D': '#d9838d'   # Soft Coral Rose / Grade D ("Hazardous")
}

HOLC_EDGE_COLORS = {
    'A': '#2b5e28',
    'B': '#256372',
    'C': '#8a6d10',
    'D': '#8a2b34'
}

HOLC_NAMES = {
    'A': 'Grade A ("Best")',
    'B': 'Grade B ("Still Desirable")',
    'C': 'Grade C ("Definitely Declining")',
    'D': 'Grade D ("Hazardous")'
}


def tree_bbox(holc_gdf, margin=0.005):
    """Bounding box covering the HOLC polygons and the Figure 1 panel (b) window."""
    minx_h, miny_h, maxx_h, maxy_h = holc_gdf.total_bounds
    center_lat = (miny_h + maxy_h) / 2.0
    span_lon = PANEL_B_SPAN_LAT * PANEL_B_ASPECT / np.cos(np.radians(30.288))
    minx = min(minx_h, PANEL_B_CENTER_LON - span_lon / 2.0) - margin
    maxx = max(maxx_h, PANEL_B_CENTER_LON + span_lon / 2.0) + margin
    miny = min(miny_h, center_lat - PANEL_B_SPAN_LAT / 2.0) - margin
    maxy = max(maxy_h, center_lat + PANEL_B_SPAN_LAT / 2.0) + margin
    return (minx, miny, maxx, maxy)


def load_data():
    print("Loading spatial datasets...")
    repro_data.stage_redlining()
    repro_data.ensure_boundaries("austin")

    holc_gdf = gpd.read_file(HOLC_PATH)
    holc_utm = holc_gdf.to_crs(epsg=32614)
    holc_gdf['area_m2'] = holc_utm.geometry.area
    holc_gdf['area_km2'] = holc_gdf['area_m2'] / 1e6

    bnd_gdf = gpd.read_file(EXACT_BND_PATH)
    bg_gdf = gpd.read_file(EXACT_BG_PATH)

    # Latitude-corrected Austin trees inside the study window only, rebuilt
    # from the committed point data if the corrected file is not on disk.
    print("Reading Austin trees within the HOLC / panel (b) window...")
    central_trees = repro_data.load_corrected_points("austin", bbox=tree_bbox(holc_gdf))
    if central_trees.crs is None:
        central_trees = central_trees.set_crs(epsg=4326)
    print(f"  {len(central_trees):,} trees loaded")

    return holc_gdf, bnd_gdf, bg_gdf, central_trees

def compute_metrics(holc_gdf, central_trees):
    print("Computing polygon metrics...")
    # Spatial join in EPSG:4326
    joined = gpd.sjoin(central_trees, holc_gdf[['area_id', 'grade', 'geometry']], how='inner', predicate='within')
    joined = joined[~joined.index.duplicated(keep='first')]
    
    poly_stats = joined.groupby('area_id').agg(
        tree_count=('crown_area_m2', 'count'),
        total_crown_m2=('crown_area_m2', 'sum'),
        mean_crown_m2=('crown_area_m2', 'mean'),
        median_crown_m2=('crown_area_m2', 'median')
    ).reset_index()
    
    poly_df = pd.merge(holc_gdf[['area_id', 'grade', 'area_km2', 'area_m2']], poly_stats, on='area_id', how='left')
    poly_df['tree_count'] = poly_df['tree_count'].fillna(0).astype(int)
    poly_df['total_crown_m2'] = poly_df['total_crown_m2'].fillna(0.0)
    poly_df['tree_density_km2'] = poly_df['tree_count'] / poly_df['area_km2']
    poly_df['mean_crown_m2'] = np.where(poly_df['tree_count'] > 0, poly_df['total_crown_m2'] / poly_df['tree_count'], np.nan)
    poly_df['summed_crown_area_fraction_pct'] = (poly_df['total_crown_m2'] / poly_df['area_m2']) * 100.0
    poly_df = poly_df.sort_values(['grade', 'area_km2'], ascending=[True, False]).reset_index(drop=True)
    
    # Grade level aggregates
    grade_rows = []
    for g, g_df in poly_df.groupby('grade'):
        tot_area = g_df['area_km2'].sum()
        tot_trees = g_df['tree_count'].sum()
        tot_crown = g_df['total_crown_m2'].sum()
        w_dens = tot_trees / tot_area
        w_crown_frac = (tot_crown / (tot_area * 1e6)) * 100.0
        w_mean_crown = tot_crown / tot_trees if tot_trees > 0 else np.nan
        
        grade_rows.append({
            'grade': g,
            'grade_label': HOLC_NAMES[g],
            'polygon_count': len(g_df),
            'total_area_km2': tot_area,
            'area_share_pct': tot_area / poly_df['area_km2'].sum() * 100.0,
            'total_trees': tot_trees,
            'tree_share_pct': tot_trees / poly_df['tree_count'].sum() * 100.0,
            'weighted_tree_density_km2': w_dens,
            'unweighted_mean_density_km2': g_df['tree_density_km2'].mean(),
            'weighted_mean_crown_m2': w_mean_crown,
            'unweighted_mean_crown_m2': g_df['mean_crown_m2'].mean(),
            'weighted_crown_fraction_pct': w_crown_frac,
            'unweighted_mean_crown_fraction_pct': g_df['summed_crown_area_fraction_pct'].mean()
        })
    grade_df = pd.DataFrame(grade_rows)
    
    # Sensitivity & LOPO analysis
    lopo_rows = []
    for g, g_df in poly_df.groupby('grade'):
        tot_area_full = g_df['area_km2'].sum()
        tot_trees_full = g_df['tree_count'].sum()
        tot_crown_full = g_df['total_crown_m2'].sum()
        dens_full = tot_trees_full / tot_area_full
        canopy_full = (tot_crown_full / (tot_area_full * 1e6)) * 100.0
        n_poly = len(g_df)
        
        for _, r in g_df.iterrows():
            p_id = r['area_id']
            if n_poly == 1:
                lopo_rows.append({
                    'area_id': p_id,
                    'grade': g,
                    'polygon_area_km2': r['area_km2'],
                    'polygon_trees': r['tree_count'],
                    'grade_n_polygons': 1,
                    'full_grade_density': dens_full,
                    'lopo_grade_density': np.nan,
                    'delta_density': np.nan,
                    'pct_change_density': np.nan,
                    'full_grade_canopy_pct': canopy_full,
                    'lopo_grade_canopy_pct': np.nan,
                    'delta_canopy_pct': np.nan,
                    'pct_change_canopy': np.nan,
                    'status': 'Undefined (N=1)'
                })
            else:
                sub = g_df[g_df['area_id'] != p_id]
                tot_area_sub = sub['area_km2'].sum()
                tot_trees_sub = sub['tree_count'].sum()
                tot_crown_sub = sub['total_crown_m2'].sum()
                dens_sub = tot_trees_sub / tot_area_sub
                canopy_sub = (tot_crown_sub / (tot_area_sub * 1e6)) * 100.0
                
                delta_d = dens_sub - dens_full
                pct_d = (delta_d / dens_full) * 100.0
                delta_c = canopy_sub - canopy_full
                pct_c = (delta_c / canopy_full) * 100.0
                
                lopo_rows.append({
                    'area_id': p_id,
                    'grade': g,
                    'polygon_area_km2': r['area_km2'],
                    'polygon_trees': r['tree_count'],
                    'grade_n_polygons': n_poly,
                    'full_grade_density': dens_full,
                    'lopo_grade_density': dens_sub,
                    'delta_density': delta_d,
                    'pct_change_density': pct_d,
                    'full_grade_canopy_pct': canopy_full,
                    'lopo_grade_canopy_pct': canopy_sub,
                    'delta_canopy_pct': delta_c,
                    'pct_change_canopy': pct_c,
                    'status': 'Calculated'
                })
    lopo_df = pd.DataFrame(lopo_rows)
    
    # Save CSV tables
    poly_df.to_csv(os.path.join(OUTPUT_DIR, "austin_holc_polygon_metrics.csv"), index=False)
    grade_df.to_csv(os.path.join(OUTPUT_DIR, "austin_holc_grade_metrics.csv"), index=False)
    lopo_df.to_csv(os.path.join(OUTPUT_DIR, "austin_holc_sensitivity_analysis.csv"), index=False)
    print("CSV tables exported successfully.")
    
    return poly_df, grade_df, lopo_df, joined

def generate_figure_1(holc_gdf, bnd_gdf, bg_gdf, central_trees, joined_trees):
    print("Generating Figure 1: Spatial Map (Equal Heights, Widened Plot B, Inside Legend)...")
    fig_w = 13.5
    fig_h = 7.0
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)
    
    bottom_norm = 0.13
    height_norm = 0.76
    H_in = height_norm * fig_h  # 5.32 inches
    
    # ── Panel (a): Municipal Footprint Overview ──
    cos_lat_a = np.cos(np.radians(30.296))
    minx_a, miny_a, maxx_a, maxy_a = bnd_gdf.total_bounds
    span_lat_a = 0.490
    center_lat_a = (miny_a + maxy_a) / 2.0
    W_in_a = 4.35  # inches
    width_norm_a = W_in_a / fig_w
    left_norm_a = 0.045
    
    span_lon_a = span_lat_a * (W_in_a / H_in) / cos_lat_a
    center_lon_a = (minx_a + maxx_a) / 2.0
    
    ax_a = fig.add_axes([left_norm_a, bottom_norm, width_norm_a, height_norm])
    ax_a.set_facecolor('#ffffff')
    
    # Plot layers in Panel A
    bg_gdf.plot(ax=ax_a, facecolor='#f8f9fa', edgecolor='#e2e8f0', linewidth=0.35, zorder=1)
    bnd_gdf.plot(ax=ax_a, facecolor='none', edgecolor='#212529', linewidth=1.2, zorder=3)
    for g, g_df in holc_gdf.groupby('grade'):
        g_df.plot(ax=ax_a, facecolor=HOLC_COLORS[g], edgecolor=HOLC_EDGE_COLORS[g], linewidth=0.8, alpha=0.9, zorder=4)
        
    # Zoom bounding box for study area (clean red dashed outline without messy floating text)
    minx_h, miny_h, maxx_h, maxy_h = holc_gdf.total_bounds
    pad_x_h = (maxx_h - minx_h) * 0.16
    pad_y_h = (maxy_h - miny_h) * 0.16
    zoom_box = box(minx_h - pad_x_h, miny_h - pad_y_h, maxx_h + pad_x_h, maxy_h + pad_y_h)
    zx, zy = zoom_box.exterior.xy
    ax_a.plot(zx, zy, color='#c92a2a', linestyle='--', linewidth=1.4, zorder=6)
    
    ax_a.set_title("(a) Austin municipal overview", loc='left', pad=12, fontsize=13.5, fontweight='bold')
    ax_a.set_xlim(center_lon_a - span_lon_a / 2.0, center_lon_a + span_lon_a / 2.0)
    ax_a.set_ylim(center_lat_a - span_lat_a / 2.0, center_lat_a + span_lat_a / 2.0)
    ax_a.set_aspect('equal')
    ax_a.set_xticks([])
    ax_a.set_yticks([])
    ax_a.set_xlabel('')  # newer geopandas adds CRS axis labels by default
    ax_a.set_ylabel('')
    # Remove all boundary spines on Panel A
    for spine in ax_a.spines.values():
        spine.set_visible(False)
        
    # Small neat legend in Panel A identifying the red dashed box
    legend_elements_a = [
        mlines.Line2D([], [], color='#c92a2a', linestyle='--', linewidth=1.4, label='Historical study area')
    ]
    ax_a.legend(handles=legend_elements_a, loc='lower right', bbox_to_anchor=(0.98, 0.04),
                frameon=False, fontsize=10.5)
        
    # Scale bar in Panel A (10 km) - clean bar without end ticks
    mid_lat_a = center_lat_a
    km_per_deg_lon_a = 111.32 * np.cos(np.radians(mid_lat_a))
    deg_len_10km = 10.0 / km_per_deg_lon_a
    sb_xa = center_lon_a - span_lon_a * 0.42
    sb_ya = center_lat_a - span_lat_a * 0.44
    ax_a.plot([sb_xa, sb_xa + deg_len_10km], [sb_ya, sb_ya], color='#212529', lw=1.3, solid_capstyle='butt', zorder=8)
    ax_a.text(sb_xa + deg_len_10km / 2.0, sb_ya + span_lat_a * 0.010, "10 km", ha='center', va='bottom', fontsize=10.0, color='#212529', zorder=8)
    
    # ── Panel (b): Central Historical Core Detail (WIDENED: 7.00 inches wide) ──
    cos_lat_b = np.cos(np.radians(30.288))
    span_lat_b = 0.142
    center_lat_b = (miny_h + maxy_h) / 2.0
    W_in_b = 7.00  # Wide! 7.00 inches vs H_in = 5.32 inches (aspect ratio 1.316)
    width_norm_b = W_in_b / fig_w
    left_norm_b = left_norm_a + width_norm_a + 0.048
    
    span_lon_b = span_lat_b * (W_in_b / H_in) / cos_lat_b
    center_lon_b = -97.745
    
    ax_b = fig.add_axes([left_norm_b, bottom_norm, width_norm_b, height_norm])
    ax_b.set_facecolor('#ffffff')
    
    # 1. Historically un-graded area shown neutrally
    hull = holc_gdf.union_all().convex_hull
    ungraded_area = hull.difference(holc_gdf.union_all())
    if not ungraded_area.is_empty:
        gpd.GeoSeries([ungraded_area]).plot(ax=ax_b, facecolor='#f1f3f5', edgecolor='#ced4da', linewidth=0.7, linestyle=':', alpha=0.9, zorder=2)
        
    # 2. Contemporary detected tree locations across wide area
    ct_zoom = central_trees.cx[center_lon_b - span_lon_b/2.0:center_lon_b + span_lon_b/2.0, center_lat_b - span_lat_b/2.0:center_lat_b + span_lat_b/2.0]
    ax_b.scatter(ct_zoom.geometry.x, ct_zoom.geometry.y, s=0.20, color='#2d6a4f', alpha=0.14, edgecolors='none', zorder=3)
    
    # 3. Plot 18 HOLC Polygons with clean borders
    for g, g_df in holc_gdf.groupby('grade'):
        g_df.plot(ax=ax_b, facecolor=HOLC_COLORS[g], edgecolor=HOLC_EDGE_COLORS[g], linewidth=1.1, alpha=0.70, zorder=4)
        
    # 4. Highlight Polygon 9093 cleanly without clutter
    p9093_sub = holc_gdf[holc_gdf['area_id'] == 9093]
    p9093_sub.plot(ax=ax_b, facecolor='none', edgecolor='#143813', linewidth=2.2, zorder=6)
    
    # Unobtrusive callout for Polygon 9093
    ax_b.annotate("Polygon 9093\n(Grade A)",
                  xy=(-97.770, 30.306),
                  xytext=(-97.822, 30.328),
                  fontsize=10.0, fontweight='bold', color='#143813',
                  arrowprops=dict(arrowstyle='->', color='#143813', lw=1.3),
                  bbox=dict(boxstyle='square,pad=0.3', facecolor='#ffffff', edgecolor='#2e7d32', alpha=0.92, lw=0.8),
                  zorder=9)
                  
    # 5. Neutral road annotation for Interstate 35 corridor slightly moved up
    i35_lons = [-97.727, -97.729, -97.732, -97.736, -97.739, -97.742]
    i35_lats = [30.330, 30.295, 30.270, 30.245, 30.228, 30.218]
    ax_b.plot(i35_lons, i35_lats, color='#495057', linestyle='--', linewidth=1.3, alpha=0.75, zorder=5)
    ax_b.text(-97.733, 30.231, "Interstate 35", fontsize=10.5, fontstyle='italic', color='#495057',
              ha='left', va='center', rotation=68, zorder=8)
              
    # Scale Bar (2 km) in Panel B placed cleanly at bottom-right without end ticks
    mid_lat = center_lat_b
    km_per_deg_lon = 111.32 * np.cos(np.radians(mid_lat))
    deg_len_2km = 2.0 / km_per_deg_lon
    
    sb_x0 = center_lon_b + span_lon_b * 0.44 - deg_len_2km
    sb_y0 = center_lat_b - span_lat_b * 0.44
    
    ax_b.plot([sb_x0, sb_x0 + deg_len_2km], [sb_y0, sb_y0], color='#212529', lw=1.3, solid_capstyle='butt', zorder=12)
    ax_b.text(sb_x0 + deg_len_2km / 2.0, sb_y0 + span_lat_b * 0.010, "2 km", ha='center', va='bottom', fontsize=10.0, color='#212529', zorder=12)
    
    # North arrow - smaller, subtle, and compact
    na_x = center_lon_b + span_lon_b * 0.45
    na_y = center_lat_b + span_lat_b * 0.43
    ax_b.annotate('N', xy=(na_x, na_y + 0.005), xytext=(na_x, na_y - 0.003),
                  ha='center', va='center', fontsize=10.5, fontweight='bold', color='#212529',
                  arrowprops=dict(facecolor='#212529', edgecolor='none', width=0.8, headwidth=3.2, headlength=4.0), zorder=12)
                  
    # Axis limits & appearance: Remove boundaries, only keep x and y lines
    ax_b.set_xlim(center_lon_b - span_lon_b / 2.0, center_lon_b + span_lon_b / 2.0)
    ax_b.set_ylim(center_lat_b - span_lat_b / 2.0, center_lat_b + span_lat_b / 2.0)
    ax_b.set_aspect('equal')
    ax_b.set_title("(b) Historical HOLC study area & detected tree locations", loc='left', pad=12, fontsize=13.5, fontweight='bold')
    ax_b.set_xlabel("Longitude (°W)", fontsize=12.5, fontweight='bold')
    ax_b.set_ylabel("Latitude (°N)", fontsize=12.5, fontweight='bold')
    ax_b.tick_params(axis='both', labelsize=10.5, color='#495057')
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)
    ax_b.spines['left'].set_visible(True)
    ax_b.spines['bottom'].set_visible(True)
    ax_b.spines['left'].set_color('#495057')
    ax_b.spines['bottom'].set_color('#495057')
    ax_b.spines['left'].set_linewidth(0.9)
    ax_b.spines['bottom'].set_linewidth(0.9)
        
    # Legend placed inside Panel B above the scale bar, balanced position
    legend_elements = [
        Patch(facecolor=HOLC_COLORS['A'], edgecolor=HOLC_EDGE_COLORS['A'], label='Grade A: "Best"'),
        Patch(facecolor=HOLC_COLORS['B'], edgecolor=HOLC_EDGE_COLORS['B'], label='Grade B: "Still Desirable"'),
        Patch(facecolor=HOLC_COLORS['C'], edgecolor=HOLC_EDGE_COLORS['C'], label='Grade C: "Definitely Declining"'),
        Patch(facecolor=HOLC_COLORS['D'], edgecolor=HOLC_EDGE_COLORS['D'], label='Grade D: "Hazardous"'),
        Patch(facecolor='#f1f3f5', edgecolor='#ced4da', label='Historically un-graded area'),
        mlines.Line2D([], [], color='#2d6a4f', marker='o', linestyle='None', markersize=5.5, alpha=0.6, label='Detected tree locations')
    ]
    leg = ax_b.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.11), ncol=3,
                      frameon=False, prop={'weight': 'normal', 'size': 10.5})  # below the map: obscures no data; regular weight (author request)
    
    out_png = os.path.join(OUTPUT_DIR, "figure_1_holc_austin_tree_distribution.png")
    out_pdf = os.path.join(OUTPUT_DIR, "figure_1_holc_austin_tree_distribution.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Figure 1 to {out_png} and {out_pdf}")

def generate_figure_2(poly_df, grade_df):
    print("Generating Figure 2: HOLC Grade Comparisons (Nature Portfolio Standards)...")
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.8), dpi=300, sharex=True)
    fig.subplots_adjust(wspace=0.32, bottom=0.22, top=0.82, left=0.08, right=0.98)
    
    # Grade positions
    grades = ['A', 'B', 'C', 'D']
    x_pos = {g: i for i, g in enumerate(grades)}
    x_labels = [f"Grade {g}\n(n = {len(poly_df[poly_df['grade']==g])})" for g in grades]
    
    # Deterministic jitter generator for polygon points
    np.random.seed(42)
    jitter_map = {}
    for g in grades:
        sub = poly_df[poly_df['grade'] == g]
        if len(sub) == 1:
            jitter_map[sub.iloc[0]['area_id']] = 0.0
        else:
            jits = np.linspace(-0.15, 0.15, len(sub))
            np.random.shuffle(jits)
            for p_id, jit in zip(sub['area_id'], jits):
                jitter_map[p_id] = jit

    # Panels configuration
    panels = [
        (axes[0], 'a', 'tree_density_km2', 'weighted_tree_density_km2', 'Tree density', 'Tree density (trees / km²)'),
        (axes[1], 'b', 'mean_crown_m2', 'weighted_mean_crown_m2', 'Mean detected crown area', 'Mean crown area (m² / tree)'),
        (axes[2], 'c', 'summed_crown_area_fraction_pct', 'weighted_crown_fraction_pct', 'Summed crown-area fraction', 'Summed crown-area fraction (%)')
    ]
    
    for ax, p_let, y_col, w_col, title, ylabel in panels:
        # y-limits from the data (polygons and grade aggregates), starting at 0, 12 % headroom
        ylim = (0, 1.12 * max(poly_df[y_col].max(), grade_df[w_col].max()))
        ax.set_facecolor('#ffffff')
        ax.grid(axis='y', linestyle='--', alpha=0.35, color='#e2e8f0', linewidth=0.6, zorder=1)
        
        # Despine top and right for clean modern look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#495057')
        ax.spines['bottom'].set_color('#495057')
        ax.spines['left'].set_linewidth(0.9)
        ax.spines['bottom'].set_linewidth(0.9)
        
        # 1. Plot individual polygon observations
        for _, r in poly_df.iterrows():
            g = r['grade']
            p_id = r['area_id']
            xp = x_pos[g] + jitter_map[p_id]
            yp = r[y_col]
            
            # Subtle size scaling by polygon area
            ms = 48 + 10 * np.sqrt(r['area_km2'])
            ax.scatter(xp, yp, s=ms, facecolor=HOLC_COLORS[g], edgecolor=HOLC_EDGE_COLORS[g],
                       linewidth=1.1, alpha=0.88, zorder=4)
            
            # Only label Polygon 9093 cleanly to eliminate overlapping text
            if p_id == 9093 and y_col in ['tree_density_km2', 'summed_crown_area_fraction_pct']:
                ax.annotate("9093", (xp, yp), xytext=(0, -9), textcoords='offset points', fontsize=9.5, fontweight='bold', color='#143813', va='top', ha='center', zorder=6)

        # 2. Plot Grade-Level Aggregates (horizontal crossbar + white diamond)
        for _, gr in grade_df.iterrows():
            g = gr['grade']
            xg = x_pos[g]
            yw = gr[w_col]
            
            # Horizontal bar spanning jitter
            ax.hlines(yw, xg - 0.22, xg + 0.22, color='#1f2328', linewidth=2.0, zorder=5)
            # Distinctive white diamond
            ax.scatter(xg, yw, marker='D', s=75, facecolor='#ffffff', edgecolor='#1f2328',
                       linewidth=1.8, zorder=6)

        ax.set_xticks(range(len(grades)))
        ax.set_xticklabels(x_labels, fontsize=11.0, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12.5, fontweight='bold')
        ax.tick_params(axis='y', labelsize=10.5)
        ax.set_ylim(ylim)
        ax.set_xlim(-0.5, 3.5)
        
        # Bold panel title with parentheses
        ax.set_title(f"({p_let}) {title}", loc='left', pad=12, fontsize=13.0, fontweight='bold')

    # Minimal Figure Legend with larger text
    legend_handles = [
        mlines.Line2D([], [], color='#495057', marker='o', linestyle='None', markersize=8.0,
                      markeredgecolor='#1f2328', markerfacecolor='#adb5bd', label='Individual historical polygon (n = 18)'),
        mlines.Line2D([], [], color='#1f2328', marker='D', linestyle='-', linewidth=2.0, markersize=7.5,
                      markeredgecolor='#1f2328', markerfacecolor='#ffffff', label='Grade-level aggregate')
    ]
    fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 0.99),
               ncol=2, frameon=False, fontsize=10.5)

    out_png = os.path.join(OUTPUT_DIR, "figure_2_polygon_tree_metrics_by_grade.png")
    out_pdf = os.path.join(OUTPUT_DIR, "figure_2_polygon_tree_metrics_by_grade.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Figure 2 to {out_png} and {out_pdf}")

def generate_figure_3(poly_df, grade_df, lopo_df):
    # Every value is computed from this run's tables (poly_df, grade_df, lopo_df). The earlier version of this panel set
    # carried the ORIGINAL dataset's numbers as literals; replaced 2026-11-02.
    print("Generating Figure 3: Sensitivity and Influence Analysis (data-driven, 2 x 2)...")
    fig = plt.figure(figsize=(11.0, 9.6), dpi=300)
    gs = gridspec.GridSpec(2, 2, wspace=0.34, hspace=0.62, left=0.08, right=0.985, top=0.94, bottom=0.10)
    def style(ax, axis='y'):
        ax.set_facecolor('#ffffff')
        ax.grid(axis=axis, linestyle='--', alpha=0.35, color='#e2e8f0', linewidth=0.6, zorder=1)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        for sp in ('left', 'bottom'):
            ax.spines[sp].set_color('#495057'); ax.spines[sp].set_linewidth(1.0)

    p9093 = poly_df[poly_df['area_id'] == 9093].iloc[0]
    gr_a = grade_df[grade_df['grade'] == 'A'].iloc[0]
    gr_b = grade_df[grade_df['grade'] == 'B'].iloc[0]
    lo = lopo_df[lopo_df['area_id'] == 9093].iloc[0]
    others = ', '.join(str(int(x)) for x in poly_df[(poly_df['grade'] == 'A') & (poly_df['area_id'] != 9093)]['area_id'])

    # ── (a) Polygon 9093 share of Grade A ──
    ax_a = fig.add_subplot(gs[0, 0]); style(ax_a)
    area_pct = 100.0 * p9093['area_km2'] / gr_a['total_area_km2']
    trees_pct = 100.0 * p9093['tree_count'] / gr_a['total_trees']
    categories = [f"Land area\n({gr_a['total_area_km2']:.2f} km²)", f"Detected trees\n({int(gr_a['total_trees']):,} trees)"]
    p_sh = [area_pct, trees_pct]; o_sh = [100.0 - area_pct, 100.0 - trees_pct]
    x_idx = np.arange(2)
    ax_a.bar(x_idx, p_sh, width=0.48, color='#2e7d32', edgecolor='#1b5e20', linewidth=1.1, label='Polygon 9093', zorder=3)
    ax_a.bar(x_idx, o_sh, width=0.48, bottom=p_sh, color='#a5d6a7', edgecolor='#2e7d32', linewidth=1.1,
             label=f'Remaining Grade A ({others})', zorder=3)
    ax_a.text(0, area_pct / 2, f"{area_pct:.1f}%\n({p9093['area_km2']:.2f} km²)", ha='center', va='center', color='#ffffff', fontweight='bold', fontsize=11)
    ax_a.text(0, area_pct + o_sh[0] / 2, f"{o_sh[0]:.1f}%\n({gr_a['total_area_km2'] - p9093['area_km2']:.2f} km²)", ha='center', va='center', color='#1b5e20', fontweight='bold', fontsize=10.5)
    ax_a.text(1, trees_pct / 2, f"{trees_pct:.1f}%\n({int(p9093['tree_count']):,})", ha='center', va='center', color='#ffffff', fontweight='bold', fontsize=11)
    ax_a.text(1, trees_pct + o_sh[1] / 2, f"{o_sh[1]:.1f}%\n({int(gr_a['total_trees'] - p9093['tree_count']):,})", ha='center', va='center', color='#1b5e20', fontweight='bold', fontsize=10.5)
    ax_a.set_xticks(x_idx); ax_a.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax_a.set_ylabel("Share of Grade A (%)", fontsize=12.5, fontweight='bold'); ax_a.tick_params(axis='y', labelsize=11)
    ax_a.set_ylim(0, 105); ax_a.set_xlim(-0.6, 1.6)
    ax_a.set_title("(a) Polygon 9093 share of Grade A", loc='left', pad=12, fontsize=13, fontweight='bold')
    ax_a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=2, frameon=False, fontsize=10)

    scenarios = ["Grade A\n(all 3)", "Polygon 9093\nalone", "Grade A\n(excl. 9093)", "Grade B\n(reference)"]
    colors = ['#76a865', '#2e7d32', '#e57373', '#7cb5bd']; edges = ['#2b5e28', '#1b5e20', '#b71c1c', '#256372']
    x_b = np.arange(4)
    for pos, let, vals, ref, ylab, ttl, fmt in [
            (gs[0, 1], 'b', [gr_a['weighted_tree_density_km2'], p9093['tree_density_km2'], lo['lopo_grade_density'], gr_b['weighted_tree_density_km2']],
             gr_b['weighted_tree_density_km2'], "Density (trees / km²)", "Tree density under exclusion scenarios", "{:,.0f}"),
            (gs[1, 0], 'c', [gr_a['weighted_crown_fraction_pct'], p9093['summed_crown_area_fraction_pct'], lo['lopo_grade_canopy_pct'], gr_b['weighted_crown_fraction_pct']],
             gr_b['weighted_crown_fraction_pct'], "Crown fraction (%)", "Summed crown-area fraction scenarios", "{:.2f}%")]:
        ax = fig.add_subplot(pos); style(ax)
        top = 1.18 * max(vals)
        bars = ax.bar(x_b, vals, width=0.52, color=colors, edgecolor=edges, linewidth=1.1, zorder=3)
        ax.axhline(ref, color='#0277bd', linestyle=':', linewidth=1.4, zorder=2)
        ax.text(0.99, 1.0, "Grade B baseline: " + fmt.format(ref), transform=ax.transAxes, color='#0277bd',
                fontsize=10.5, ha='right', va='bottom', fontweight='bold')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015 * top, fmt.format(v), ha='center', va='bottom', fontsize=10.5, fontweight='bold', color='#1f2328')
        ax.set_xticks(x_b); ax.set_xticklabels(scenarios, fontsize=10.5, fontweight='bold')
        ax.set_ylabel(ylab, fontsize=12.5, fontweight='bold'); ax.tick_params(axis='y', labelsize=11)
        ax.set_ylim(0, top)
        ax.set_title(f"({let}) {ttl}", loc='left', pad=22, fontsize=13, fontweight='bold')

    # ── (d) Leave-one-polygon-out ──
    ax_d = fig.add_subplot(gs[1, 1]); style(ax_d, axis='x')
    lv = lopo_df.dropna(subset=['pct_change_density']).sort_values('pct_change_density').copy()
    y = np.arange(len(lv))
    ax_d.barh(y, lv['pct_change_density'], height=0.65, color=[HOLC_COLORS[g] for g in lv['grade']],
              edgecolor=[HOLC_EDGE_COLORS[g] for g in lv['grade']], linewidth=0.9, zorder=3)
    ax_d.axvline(0, color='#1f2328', linewidth=1.2, zorder=2)
    ax_d.set_yticks(y); ax_d.set_yticklabels([f"{int(a)} ({g})" for a, g in zip(lv['area_id'], lv['grade'])], fontsize=10)
    ax_d.set_ylabel("Omitted historical polygon (HOLC grade)", fontsize=12, fontweight='bold')
    ax_d.set_xlabel("Change in grade tree density when omitted (%)", fontsize=12, fontweight='bold')
    ax_d.tick_params(axis='x', labelsize=11)
    lim = 1.15 * max(abs(lv['pct_change_density'].min()), abs(lv['pct_change_density'].max()))
    ax_d.set_xlim(-lim, lim)
    undefined = lopo_df[lopo_df['pct_change_density'].isna()]
    if len(undefined):
        u = undefined.iloc[0]
        ax_d.text(0.98, 0.04, f"* Grade {u['grade']} ({int(u['area_id'])}) LOPO\n  undefined (n = 1)", transform=ax_d.transAxes,
                  fontsize=10, ha='right', va='bottom', bbox=dict(boxstyle='square,pad=0.35', facecolor='#fffde7', edgecolor='#d4ac0d', alpha=0.9, lw=0.8))
    ax_d.set_title("(d) Leave-one-polygon-out sensitivity", loc='left', pad=12, fontsize=13, fontweight='bold')

    out_png = os.path.join(OUTPUT_DIR, "figure_3_influence_sensitivity_analysis.png")
    out_pdf = os.path.join(OUTPUT_DIR, "figure_3_influence_sensitivity_analysis.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Figure 3 to {out_png} and {out_pdf}")

def main():
    print("=" * 80)
    print("STARTING NATURE PORTFOLIO SCIENTIFIC FIGURE GENERATION")
    print("=" * 80)
    holc_gdf, bnd_gdf, bg_gdf, central_trees = load_data()
    poly_df, grade_df, lopo_df, joined_trees = compute_metrics(holc_gdf, central_trees)
    
    generate_figure_1(holc_gdf, bnd_gdf, bg_gdf, central_trees, joined_trees)
    generate_figure_2(poly_df, grade_df)
    generate_figure_3(poly_df, grade_df, lopo_df)
    print("=" * 80)
    print("ALL REFINED FIGURES SUCCESSFULLY GENERATED IN:", OUTPUT_DIR)
    print("=" * 80)

if __name__ == '__main__':
    main()
