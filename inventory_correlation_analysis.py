"""
Sherlock script to run Spearman correlation analysis between public inventory tree counts 
and model-predicted tree counts at the census block group level, across four cities.

Produces two distinct publication-ready figures for Nature Cities:
1. `_combined`: Single panel with all four cities overlaid.
2. `_4panel`: 2x2 subplot grid with bolded city titles and updated y-axis limits.

Refinements:
- X-axis uses log10(x+1) transformation with horizontal jitter for zero counts.
- Axes: x [0, 10K], y [1K, 200K].
- Spearman correlation computed on raw untransformed counts including zeros.
- Combined figure: Upper right legend and expanded per-city stats block.
"""

import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for Sherlock
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from scipy import stats

# ============================================================================
# CONFIGURATION & STYLE (Nature Cities Standard)
# ============================================================================
BASE_DIR = os.environ.get("TREE_MAPPING_BASE_DIR", os.getcwd())
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

INVENTORY_CSV = os.path.join(BASE_DIR, "urbantreedata_2023/new_data.csv")
# Updated to use the standardized Ripley point datasets
RIPLEY_DATA_DIR = os.path.join(BASE_DIR, "analysis_output", "ripley_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_output/inventory_correlation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Standardized UTM projections (meters) from prepare_ripley_data.py
CITY_UTM_EPSG = {
    'austin':      32614,   # UTM 14N
    'bloomington': 32616,   # UTM 16N
    'cupertino':   32610,   # UTM 10N
    'surrey':      32610,   # UTM 10N
}

CITIES = ['austin', 'bloomington', 'cupertino', 'surrey']
GRID_POS = {'austin': (0, 0), 'bloomington': (0, 1), 'cupertino': (1, 0), 'surrey': (1, 1)}
PANEL_LABELS = {'austin': 'a', 'bloomington': 'b', 'cupertino': 'c', 'surrey': 'd'}

# New Exact Boundaries Configuration
EXACT_BOUNDARIES_DIR = os.path.join(BASE_DIR, 'census_boundaries', 'exact_boundaries')

CITY_LABELS = {
    'austin': 'Austin, TX', 'bloomington': 'Bloomington, IN',
    'cupertino': 'Cupertino, CA', 'surrey': 'Surrey, BC',
}

OI_COLORS = {
    'austin':      '#E69F00', 'bloomington': '#56B4E9',
    'cupertino':   '#009E73', 'surrey':      '#CC79A7',
}

FONT_FAMILY = 'Arial'
LABEL_FONT_SIZE = 10
TICK_FONT_SIZE = 8.5
PANEL_LABEL_SIZE = 12
MARKER_SIZE = 7
MARKER_ALPHA = 0.55
REG_LINE_WEIGHT = 2.0
REG_CI_ALPHA_DEFAULT = 0.20
COMBINED_CI_ALPHA = 0.20
ANNOTATION_FONT_SIZE = 7

AXIS_LIMITS_X_TRANS = (0, np.log10(10000 + 1))
AXIS_LIMITS_Y_TRANS = (0, np.log10(300000 + 1))

plt.rcParams.update({
    'font.family': FONT_FAMILY,
    'font.size': 8,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'savefig.dpi': 300,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'mathtext.fontset': 'custom',
    'mathtext.rm': FONT_FAMILY,
    'mathtext.it': f'{FONT_FAMILY}:italic'
})

def log_tick_formatter(x, pos):
    """Format log ticks with K notation."""
    if x < 1000: return f'{int(x)}'
    if x < 1000000: return f'{int(x/1000)}K'
    return f'{int(x/1000000)}M'

def to_superscript(s):
    """Convert a string or number to Unicode superscripts."""
    trans = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
    return str(s).translate(trans)

def get_significance_stars(p):
    """Return significance stars based on p-value."""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

def format_p_value_unicode(p):
    """Format p-value using Unicode multiplication and superscripts."""
    if p < 0.001:
        exp = int(np.floor(np.log10(p)))
        base = p / 10**exp
        return f"{base:.1f} \u00D7 10{to_superscript(exp)}"
    return f"{p:.3f}"

def format_stats_line(city_label, rho, p, n):
    """Build a unified stats line for the legend."""
    stars = get_significance_stars(p)
    p_str = format_p_value_unicode(p)
    # Using mathtext for italic variables, but keeping p-value block as requested
    return f"{city_label} $\\mathit{{\\rho}}$ = {rho:.2f}{stars} ($\\mathit{{p}}$ = {p_str}, $\\mathit{{n}}$ = {n})"

def collect_city_data(df_inventory_all):
    """Analytics collection for all cities using exact boundaries and Ripley points."""
    city_data = {}
    for city in CITIES:
        print(f"  Collecting {city.title()}...")
        
        # 1. Load Exact Census Boundaries
        census_path = os.path.join(EXACT_BOUNDARIES_DIR, f"{city.lower()}_exact_block_groups.geojson")
        if not os.path.exists(census_path):
            print(f"    Warning: Census path {census_path} not found.")
            continue
            
        gdf_census = gpd.read_file(census_path)
        if gdf_census.crs != 'EPSG:4326': gdf_census = gdf_census.to_crs('EPSG:4326')
        
        # Robustly detect GEOID column and clean it
        geoid_col = next((c for c in ('GEOID', 'census_id', 'DAUID', 'GEO_ID') if c in gdf_census.columns), 'index')
        
        def clean_geoid(val):
            s = str(val).strip()
            if 'US' in s:
                s = s.split('US')[-1]
            return s
            
        gdf_census['clean_geoid'] = gdf_census[geoid_col].apply(clean_geoid)
        
        # Determine the target block group ID length (12 for US census block group, 8 for CA DA)
        bg_len = 8 if city.lower() == 'surrey' else 12
        gdf_census['BG_GEOID'] = gdf_census['clean_geoid'].str.slice(0, bg_len)
        
        # 2. Load Model Points (Ripley GeoJSON format) and Aggregate
        ripley_geojson = os.path.join(BASE_DIR, "analysis_output", "ripley_data", f"{city}_ripley_points_with_income.geojson")
        if not os.path.exists(ripley_geojson):
            # Fallback for local testing
            ripley_geojson = os.path.join(
                os.environ.get("TREE_MAPPING_TREES_DIR", os.path.join(BASE_DIR, "all_trees")),
                f"{city}_ripley_points_with_income.geojson",
            )
            
        if not os.path.exists(ripley_geojson):
            print(f"    Warning: Model GeoJSON {ripley_geojson} not found.")
            continue
            
        print(f"    Loading model trees from GeoJSON...")
        gdf_model_pts = gpd.read_file(ripley_geojson)
        if gdf_model_pts.crs != "EPSG:4326":
            gdf_model_pts = gdf_model_pts.to_crs("EPSG:4326")
        
        geo_unit_name = "dissemination areas" if city.lower() == 'surrey' else "census block groups"
        n_unique_bg = len(gdf_census['BG_GEOID'].unique())
        print(f"    Aggregating {len(gdf_model_pts):,} model trees via exact spatial join into {n_unique_bg} {geo_unit_name}...")
        joined_model = gpd.sjoin(gdf_model_pts, gdf_census[[geoid_col, 'BG_GEOID', 'geometry']], how='inner', predicate='intersects')
        joined_model = joined_model[~joined_model.index.duplicated(keep='first')]
        model_counts = joined_model.groupby('BG_GEOID').size().reset_index(name='tree_count')
        model_counts.columns = ['GEOID', 'tree_count']
        model_counts['GEOID'] = model_counts['GEOID'].astype(str)
        
        # 3. Load Inventory Points and Aggregate
        df_inv_city = df_inventory_all[df_inventory_all['city'].str.lower() == city.lower()].copy()
        if df_inv_city.empty:
            inv_counts = pd.DataFrame(columns=['GEOID', 'inventory_count'])
        else:
            gdf_inv = gpd.GeoDataFrame(df_inv_city, geometry=gpd.points_from_xy(df_inv_city.long, df_inv_city.lat), crs="EPSG:4326")
            joined_inv = gpd.sjoin(gdf_inv, gdf_census[[geoid_col, 'BG_GEOID', 'geometry']], how='inner', predicate='intersects')
            # 1:1 assignment
            if not joined_inv.empty:
                joined_inv = joined_inv[~joined_inv.index.duplicated(keep='first')]
                inv_counts = joined_inv.groupby('BG_GEOID').size().reset_index(name='inventory_count')
                inv_counts.columns = ['GEOID', 'inventory_count']
            else:
                inv_counts = pd.DataFrame(columns=['GEOID', 'inventory_count'])
            
        # 4. Merge and Filter (Ensure all census units are included)
        df_merged = model_counts.merge(inv_counts, on='GEOID', how='outer').fillna(0)
        valid_geoids = set(gdf_census['BG_GEOID'])
        
        df_plot_base = pd.DataFrame({'GEOID': list(valid_geoids)})
        df_plot = df_plot_base.merge(df_merged, on='GEOID', how='left').fillna(0)
        
        rho, p_val = stats.spearmanr(df_plot['inventory_count'], df_plot['tree_count'])
        n = len(df_plot)
        
        x_raw = df_plot['inventory_count'].values
        y_raw = df_plot['tree_count'].values
        x_log_trans = np.log10(x_raw + 1.0)
        y_log_trans = np.log10(y_raw + 1.0)
        
        # Regression line computation (Log-Log on transformed space)
        if len(x_raw) >= 5:
            slope, intercept = np.polyfit(x_log_trans, y_log_trans, 1)
            x_eval_trans = np.linspace(x_log_trans.min(), x_log_trans.max(), 100)
        else:
            slope, intercept, x_eval_trans = 0, 0, np.array([])
        
        # Horizontal jitter for inventory zeros
        x_trans_display = x_log_trans.copy()
        zero_mask_x = (x_raw == 0)
        if np.any(zero_mask_x):
            x_trans_display[zero_mask_x] += np.random.uniform(-0.025, 0.025, size=np.sum(zero_mask_x))
            
        # Vertical jitter for model zeros (standardization)
        y_trans_display = y_log_trans.copy()
        zero_mask_y = (y_raw == 0)
        if np.any(zero_mask_y):
            y_trans_display[zero_mask_y] += np.random.uniform(-0.025, 0.025, size=np.sum(zero_mask_y))
            
        city_data[city] = {
            'df_plot': df_plot, 'rho': rho, 'p_val': p_val, 'n': n,
            'slope': slope, 'intercept': intercept, 'x_eval_trans': x_eval_trans,
            'x_trans_display': x_trans_display, 'y_trans_display': y_trans_display,
            'x_log_trans': x_log_trans, 'y_log_trans': y_log_trans
        }
    return city_data

def plot_regression_and_ci(ax, data, color, alpha_ci, max_val_trans=np.log10(300000+1)):
    """Reusable regression and CI band plotter on transformed scale."""
    x_eval = data['x_eval_trans']
    if len(x_eval) == 0: return
    
    y_eval_trans = data['intercept'] + data['slope'] * x_eval
    
    # Clip to display limits
    mask = (y_eval_trans >= AXIS_LIMITS_Y_TRANS[0]) & (y_eval_trans <= max_val_trans)
    if np.any(mask):
        ax.plot(x_eval[mask], y_eval_trans[mask], color=color, lw=REG_LINE_WEIGHT, zorder=5)
        
        boots_y_trans = []
        x_log = data['x_log_trans']
        y_log = data['y_log_trans']
        
        for _ in range(1000):
            idx = np.random.choice(len(x_log), len(x_log), replace=True)
            s, i = np.polyfit(x_log[idx], y_log[idx], 1)
            boots_y_trans.append(i + s * x_eval)
            
        boots_y_trans = np.array(boots_y_trans)
        low_y = np.percentile(boots_y_trans, 2.5, axis=0)
        high_y = np.percentile(boots_y_trans, 97.5, axis=0)
        ax.fill_between(x_eval[mask], np.clip(low_y[mask], AXIS_LIMITS_Y_TRANS[0], max_val_trans), 
                        np.clip(high_y[mask], AXIS_LIMITS_Y_TRANS[0], max_val_trans), 
                        color=color, alpha=alpha_ci, lw=0, zorder=4)

def setup_axes_styling(ax, is_combined=False, max_val_trans=np.log10(300000+1)):
    """Standardize axes limits and ticks using log10(val+1) scale."""
    ax.set_xlim(AXIS_LIMITS_X_TRANS)
    ax.set_ylim(AXIS_LIMITS_Y_TRANS[0], max_val_trans)
    ax.tick_params(axis='both', which='both', direction='in', labelsize=TICK_FONT_SIZE)
    
    # X ticks (inventory counts)
    x_ticks_raw = [0, 10, 100, 1000, 10000]
    ax.set_xticks(np.log10(np.array(x_ticks_raw) + 1))
    ax.set_xticklabels(['0', '10', '100', '1K', '10K'])
    
    # Y ticks (model tree counts)
    y_ticks_raw = [0, 100, 1000, 10000, 100000]
    if max_val_trans > np.log10(100000): 
        y_ticks_raw.append(int(10**max_val_trans))
    
    ax.set_yticks(np.log10(np.array(y_ticks_raw) + 1))
    ax.set_yticklabels([log_tick_formatter(y, None) for y in y_ticks_raw])
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

def main():
    print(f"Starting Spearman Correlation Final Refinements...")
    if not os.path.exists(INVENTORY_CSV):
        print("Error: Inventory CSV not found"); return

    df_inventory_all = pd.read_csv(INVENTORY_CSV, usecols=['city', 'long', 'lat'])
    city_data = collect_city_data(df_inventory_all)

    # Determine Y-Axis Limit dynamically (transformed scale)
    all_y_max_trans = []
    for city, data in city_data.items():
        all_y_max_trans.append(data['y_log_trans'].max())
    global_max_y_trans = max(all_y_max_trans)
    # Scale limit logic
    if global_max_y_trans > np.log10(300000):
        AY_MAX_TRANS = np.log10(1000000 + 1)
    else:
        AY_MAX_TRANS = np.log10(300000 + 1)
    print(f"  Setting Y-Axis Upper Limit (trans): {AY_MAX_TRANS:.2f} (Data Max Trans: {global_max_y_trans:.2f})")

    # --- FIGURE 1: 4-PANEL GRID ---
    print("\nGenerating 4-Panel Figure...")
    fig4, axes4 = plt.subplots(2, 2, figsize=(6.8, 6.8))
    
    for city in CITIES:
        if city not in city_data: continue
        row, col = GRID_POS[city]; ax = axes4[row, col]; data = city_data[city]; color = OI_COLORS[city]
        ax.scatter(data['x_trans_display'], data['y_trans_display'], s=MARKER_SIZE, alpha=MARKER_ALPHA, color=color, edgecolors='white', linewidths=0.2, zorder=3)
        plot_regression_and_ci(ax, data, color, 0.15 if data['n'] < 100 else REG_CI_ALPHA_DEFAULT, max_val_trans=AY_MAX_TRANS)
        setup_axes_styling(ax, max_val_trans=AY_MAX_TRANS)
        ax.set_title(CITY_LABELS[city], fontsize=LABEL_FONT_SIZE, fontweight='bold', pad=10)
        ax.text(0.02, 0.95, PANEL_LABELS[city], transform=ax.transAxes, fontsize=PANEL_LABEL_SIZE, fontweight='bold', va='top', ha='left', zorder=10)
        
        # Unified legend in lower right
        stats_label = format_stats_line(CITY_LABELS[city], data['rho'], data['p_val'], data['n'])
        legend_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8, label=stats_label)
        ax.legend(handles=[legend_handle], labels=[stats_label], loc='lower right', 
                  fontsize=ANNOTATION_FONT_SIZE-0.5, frameon=False, handletextpad=0.2, borderpad=0.3)
        
    fig4.text(0.5, 0.02, 'Inventory Tree Count', ha='center', fontsize=LABEL_FONT_SIZE, fontweight='bold')
    fig4.text(0.02, 0.5, 'Model-Predicted Tree Count', va='center', rotation='vertical', fontsize=LABEL_FONT_SIZE, fontweight='bold')
    plt.tight_layout(rect=[0.05, 0.05, 0.98, 0.98])
    fig4.savefig(os.path.join(OUTPUT_DIR, "inventory_correlation_4panel.png"), dpi=300, bbox_inches='tight')
    fig4.savefig(os.path.join(OUTPUT_DIR, "inventory_correlation_4panel.pdf"), bbox_inches='tight')
    plt.close(fig4)

    # --- FIGURE 2: COMBINED SINGLE PANEL ---
    print("Generating Combined Figure...")
    figc, axc = plt.subplots(figsize=(6, 5))
    legend_handles = []
    legend_labels = []
    for city in CITIES:
        if city not in city_data: continue
        data = city_data[city]; color = OI_COLORS[city]
        axc.scatter(data['x_trans_display'], data['y_trans_display'], s=MARKER_SIZE, alpha=MARKER_ALPHA, color=color, edgecolors='white', linewidths=0.1, zorder=3)
        plot_regression_and_ci(axc, data, color, COMBINED_CI_ALPHA, max_val_trans=AY_MAX_TRANS)
        
        # Build unified legend entry
        stats_label = format_stats_line(CITY_LABELS[city], data['rho'], data['p_val'], data['n'])
        legend_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8, label=stats_label)
        legend_handles.append(legend_handle)
        legend_labels.append(stats_label)

    setup_axes_styling(axc, is_combined=True, max_val_trans=AY_MAX_TRANS)
    axc.set_xlabel('Inventory Tree Count', fontsize=LABEL_FONT_SIZE, fontweight='bold')
    axc.set_ylabel('Model-Predicted Tree Count', fontsize=LABEL_FONT_SIZE, fontweight='bold')
    
    # Unified Legend in LOWER RIGHT
    axc.legend(handles=legend_handles, labels=legend_labels, loc='lower right', 
               fontsize=ANNOTATION_FONT_SIZE, frameon=False, borderpad=0.5, labelspacing=0.5, handletextpad=0.2)

    plt.tight_layout()
    figc.savefig(os.path.join(OUTPUT_DIR, "inventory_correlation_combined.png"), dpi=300, bbox_inches='tight')
    figc.savefig(os.path.join(OUTPUT_DIR, "inventory_correlation_combined.pdf"), bbox_inches='tight')
    plt.close(figc)
    print(f"\nRefinements complete. Outputs in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
