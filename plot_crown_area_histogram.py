"""
plot_crown_area_histogram.py

Standalone script to generate a 4-panel crown area histogram.
Rewritten to load pre-calculated, deduplicated, and boundary-clipped crown areas
directly from the [city]_ripley_points_with_income.geojson datasets.
"""

import os
import sys

# ── Setup paths for Sherlock & local fallbacks ────────────────────────────
BASE_DIR = os.environ.get("TREE_MAPPING_BASE_DIR", os.getcwd())
if not os.path.exists(BASE_DIR):
    BASE_DIR = os.getcwd() # local fallback

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Output directory for all figures and graphics
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Candidate directories for finding pre-computed datasets (local + Sherlock paths)
DATA_DIRS = [
    os.path.join(BASE_DIR, "analysis_output", "ripley_data"),
    os.path.join(os.path.dirname(BASE_DIR), 'ripley_data_0429206', 'all_trees'),
    os.path.join(BASE_DIR, 'ripley_data_0429206', 'all_trees'),
    os.path.join(os.getcwd(), 'analysis_output', 'ripley_data'),
    os.path.join(os.getcwd(), 'ripley_data_0429206', 'all_trees'),
]

print(f"==========================================")
print(f"ENVIRONMENT DIAGNOSTICS (plot_crown_area_histogram.py)")
print(f"==========================================")
print(f"Current Working Directory: {os.getcwd()}")
print(f"BASE_DIR: {BASE_DIR} (exists: {os.path.exists(BASE_DIR)})")
print(f"Candidate directories searched for GeoJSONs:")
for d in DATA_DIRS:
    print(f"  - {d} (exists: {os.path.exists(d)})")
print(f"Graphics will be saved to: {OUTPUT_DIR}")
print(f"==========================================\n")

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

# Publication-quality matplotlib defaults (Nature Standard)
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 7,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.width': 0.5,
    'ytick.minor.width': 0.5,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'xtick.minor.size': 3,
    'ytick.minor.size': 3,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white'
})
PUBLIC_DPI = 600

CITIES = ['austin', 'bloomington', 'cupertino', 'surrey']


def round_to_3_sig_figs(n):
    """Round an integer to 3 significant figures."""
    if n <= 100:
        return n
    import math
    power = int(math.floor(math.log10(abs(n)))) - 2
    return int(round(n, -power))


def plot_crown_area_histogram():
    """
    Generate a 4-panel crown area histogram loading directly from the
    pre-computed [city]_ripley_points_with_income.geojson files.
    """
    city_crown_areas = {city: [] for city in CITIES}

    for city in CITIES:
        print(f"\n{'='*60}")
        print(f"Loading data for {city.upper()}")
        print(f"{'='*60}")

        geojson_name = f"{city.lower()}_ripley_points_with_income.geojson"
        geojson_path = None
        for d in DATA_DIRS:
            p = os.path.join(d, geojson_name)
            if os.path.exists(p):
                geojson_path = p
                break

        if geojson_path is None:
            print(f"  ⚠ GeoJSON file {geojson_name} not found in any candidate directory.")
            print(f"  Skipping {city}.")
            continue

        print(f"  Found file at: {geojson_path}")
        print(f"  Reading {geojson_name}...")
        try:
            gdf = gpd.read_file(geojson_path)
            if 'crown_area_px' in gdf.columns:
                areas = gdf['crown_area_px'].values
                # Filter out any NaN or non-positive values
                areas = areas[~np.isnan(areas) & (areas > 0)]
                city_crown_areas[city] = areas.tolist()
                print(f"  ✓ Successfully loaded {len(areas):,} crown areas.")
            else:
                print(f"  ⚠ 'crown_area_px' column not found in {geojson_name}!")
        except Exception as e:
            print(f"  ⚠ Failed to read {geojson_name}: {e}")

    # ── Collect all areas for histogram ──────────────────────────────────
    all_areas = []
    for city in CITIES:
        all_areas.extend(city_crown_areas[city])

    if not all_areas:
        print("\nNo crown areas found across any city. Cannot plot histogram.")
        return

    # ── Colors & Limits matched to city_tree_analysis.py ──────────────────
    OI_COLORS = {
        'austin':      '#E69F00', 'bloomington': '#56B4E9',
        'cupertino':   '#009E73', 'surrey':      '#CC79A7',
    }

    n_bins = 31
    bins = np.logspace(0, 3, n_bins)

    # ── 4-panel figure (2x2) — figsize matched to income/density scatter ─
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    print("\n" + "="*80)
    print("CROWN AREA SUMMARY STATISTICS (FROM PRE-COMPUTED DATASETS)")
    print("="*80)
    print(f"{'City':<15} | {'Total Crowns':<12} | {'Min (m2)':<8} | {'Median (m2)':<11} | {'Mean (m2)':<9} | {'Max (m2)':<9} | {'% > 100 m2':<10}")
    print("-" * 92)

    for i, city in enumerate(CITIES):
        ax = axes_flat[i]
        areas = np.array(city_crown_areas[city])

        if len(areas) > 0:
            total_crowns = len(areas)
            min_area = np.min(areas)
            median_val = np.median(areas)
            mean_val = np.mean(areas)
            max_area = np.max(areas)
            pct_gt_100 = (np.sum(areas > 100) / total_crowns) * 100

            print(f"{city.title():<15} | {total_crowns:<12,} | {min_area:<8.1f} | {median_val:<11.1f} | {mean_val:<9.1f} | {max_area:<9.1f} | {pct_gt_100:<9.1f}%")

            color = OI_COLORS.get(city.lower(), '#4C7A6B')
            weights = np.ones_like(areas) / len(areas)
            ax.hist(areas, bins=bins, weights=weights, color=color, alpha=0.75, edgecolor='none')
            ax.axvline(mean_val, color='#888888', linestyle='--', linewidth=0.8)

            rounded_n = round_to_3_sig_figs(total_crowns)

            handles = [
                mpatches.Patch(color='none'),
                Line2D([0], [0], color='#888888', linestyle='--', linewidth=0.8)
            ]
            labels = [
                f"n = {rounded_n:,}",
                f"mean = {mean_val:.1f} m$^2$"
            ]
            ax.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.0, 1.0), borderaxespad=0,
                      frameon=False, fontsize=8.5, handlelength=1.5, handletextpad=0.5)
        else:
            print(f"{city.title():<15} | {'0':<12} | {'N/A':<8} | {'N/A':<11} | {'N/A':<9} | {'N/A':<9} | {'0.0%':<10}")
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)

        ax.set_xscale('log')
        ax.set_xlim(left=10**0 * 0.8, right=10**3)
        ax.set_title(city.title(), fontsize=10, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=8.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if i % 2 == 0:
            ax.set_ylabel("Proportion", fontsize=10, fontweight='bold')
        if i >= 2:
            ax.set_xlabel(r"Crown Area (m$^{\mathbf{2}}$)", fontsize=10, fontweight='bold')

    print("-" * 92 + "\n")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'crown_area_histogram.png')
    plt.savefig(out_path, dpi=PUBLIC_DPI)
    plt.close()
    print(f"Crown area histogram saved to: {out_path}")


def plot_crown_area_income_scatter():
    """
    Create a scatterplot figure showing the relationship between median household income (x-axis)
    and mean crown area per tree (m², y-axis) at the census block group level.
    Matches the style of existing combined scatterplots.
    """
    import seaborn as sns
    import matplotlib.gridspec as gridspec
    import matplotlib.ticker as ticker
    from scipy import stats as _stats

    # Define colors and labels
    OI_COLORS = {
        'austin':      '#E69F00', 'bloomington': '#56B4E9',
        'cupertino':   '#009E73', 'surrey':      '#CC79A7',
    }
    CITY_LABELS = {
        'austin':      'Austin, TX', 'bloomington': 'Bloomington, IN',
        'cupertino':   'Cupertino, CA', 'surrey':      'Surrey, BC',
    }

    # Setup publication quality figure (Nature Standard size)
    fig = plt.figure(figsize=(7.2, 5.2))
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                           hspace=0.06, wspace=0.06)
    
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    
    legend_handles, legend_labels = [], []

    print("\n" + "="*80)
    print("INCOME VS MEAN CROWN AREA RELATIONSHIP (CENSUS BLOCK GROUP LEVEL)")
    print("="*80)

    for city in CITIES:
        geojson_name = f"{city.lower()}_ripley_points_with_income.geojson"
        geojson_path = None
        for d in DATA_DIRS:
            p = os.path.join(d, geojson_name)
            if os.path.exists(p):
                geojson_path = p
                break

        if geojson_path is None:
            print(f"  ⚠ GeoJSON file {geojson_name} not found in any candidate directory. Skipping {city} for scatterplot.")
            continue

        print(f"Loading {geojson_name} (ignoring geometry)...")
        try:
            # Load without geometry for high performance (30x speedup)
            df = gpd.read_file(geojson_path, ignore_geometry=True)
            
            # Find correct column names
            income_col = 'median_household_income' if 'median_household_income' in df.columns else 'median_income'
            bg_col = 'block_group_id' if 'block_group_id' in df.columns else 'GEOID'
            area_col = 'crown_area_px' if 'crown_area_px' in df.columns else 'crown_area'

            # Drop missing values
            df = df.dropna(subset=[income_col, area_col])
            df = df[(df[income_col] > 0) & (df[area_col] > 0)]

            # Group by block group to calculate mean crown area
            grouped = df.groupby(bg_col).agg({
                area_col: ['mean', 'count'],
                income_col: 'first'
            })
            grouped.columns = ['mean_crown_area', 'tree_count', 'median_income']
            grouped = grouped.reset_index()

            # Exclude census units with fewer than 10 detected trees
            grouped = grouped[grouped['tree_count'] >= 10]
            print(f"  {city.title()}: {len(grouped)} census units with >= 10 trees.")

            if len(grouped) < 5:
                print(f"  ⚠ Too few data points for {city.title()} (n={len(grouped)}). Skipping OLS/correlation.")
                continue

            x = grouped['median_income'].values  # Raw income values
            y = grouped['mean_crown_area'].values

            # Cap y for plotting to avoid scaling issues from outliers
            y_plot = np.clip(y, 0, 100)
            y_trans = np.log10(y_plot + 1.0)

            color = OI_COLORS.get(city.lower(), '#7F7F7F')
            label = CITY_LABELS.get(city.lower(), city.title())

            # Spearman correlation
            rho, p_val = _stats.spearmanr(x, y)
            
            def _fmt_val(v):
                if v == 0: return "0"
                return f"{v:.2g}"

            def _fmt_sig(p):
                if p < 0.001: return '***'
                if p < 0.01:  return '**'
                if p < 0.05:  return '*'
                return ' ns'

            stars = _fmt_sig(p_val)
            leg_str = f"{label} $\\rho = {_fmt_val(rho)}${stars} ($p = {_fmt_val(p_val)}, n = {len(grouped)}$)"

            # Main Scatter plot
            path = ax_main.scatter(x, y_trans, s=10, alpha=0.55, color=color, 
                                   edgecolors='white', linewidths=0.2, zorder=3)
            legend_handles.append(path)
            legend_labels.append(leg_str)

            # Marginal Distributions
            sns.kdeplot(x=x, ax=ax_top, color=color, fill=True, alpha=0.25, lw=0, bw_adjust=1.0)
            sns.kdeplot(x=x, ax=ax_top, color=color, fill=False, alpha=0.9, lw=1.0, bw_adjust=1.0)
            sns.kdeplot(y=y_trans, ax=ax_right, color=color, fill=True, alpha=0.25, lw=0, bw_adjust=1.0)
            sns.kdeplot(y=y_trans, ax=ax_right, color=color, fill=False, alpha=0.9, lw=1.0, bw_adjust=1.0)

            # Fit OLS line in log-space with 95% bootstrapped confidence intervals
            x_eval = np.linspace(x.min(), x.max(), 100)
            slope, intercept = np.polyfit(x, np.log10(y + 1.0), 1)
            z_eval = intercept + slope * x_eval
            ax_main.plot(x_eval, z_eval, color=color, lw=2.0, alpha=1.0, zorder=15)

            # Bootstrap for 95% CI (1000 iterations)
            boots_y = []
            for _ in range(1000):
                idx_b = np.random.choice(len(x), len(x), replace=True)
                xb, yb = x[idx_b], y[idx_b]
                try:
                    sb, ib = np.polyfit(xb, np.log10(yb + 1.0), 1)
                    boots_y.append(ib + sb * x_eval)
                except:
                    continue

            if boots_y:
                boots_y = np.array(boots_y)
                lower_y = np.percentile(boots_y, 2.5, axis=0)
                upper_y = np.percentile(boots_y, 97.5, axis=0)
                ax_main.fill_between(x_eval, lower_y, upper_y, color=color, 
                                     alpha=0.20, lw=0, zorder=14)

        except Exception as e:
            print(f"  ⚠ Failed to process {city} for scatterplot: {e}")

    # Clean marginal axes
    for sax in [ax_top, ax_right]:
        sax.axis('off')

    # Main axis styling
    ax_main.set_xlim(0, 250000)  # Income range in dollars
    
    y_ticks_raw = [0, 2, 5, 10, 20, 50, 100]
    ax_main.set_yticks(np.log10(np.array(y_ticks_raw) + 1))
    ax_main.set_yticklabels([str(t) for t in y_ticks_raw])
    ax_main.set_ylim(0, np.log10(100 + 1))

    ax_main.set_xlabel(r"Median Household Income ($\times$\$1,000)", fontsize=10, fontweight='bold')
    ax_main.set_ylabel(r"Mean Crown Area per Tree (m$^{\mathbf{2}}$, log$_{\mathbf{10}}$(y+1) scale)", fontsize=10, fontweight='bold')
    
    ax_main.tick_params(axis='both', which='major', labelsize=8.5)
    ax_main.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'${int(x/1000):,}k'))
    ax_main.grid(False)
    ax_main.spines['right'].set_visible(False)
    ax_main.spines['top'].set_visible(False)

    # Legend (placed in lower right area, matching combined_income_density_scatter_refined exactly)
    leg = ax_main.legend(legend_handles, legend_labels, loc='lower right', 
                         bbox_to_anchor=(0.99, 0.01),
                         frameon=True, fontsize=8.2, borderpad=0.2, labelspacing=0.25)
    leg.get_frame().set_alpha(0.7)
    leg.get_frame().set_edgecolor('none')
    for handle in leg.legend_handles:
        handle.set_sizes([30.0])
        handle.set_alpha(0.9)

    plt.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.12)

    # Save outputs
    for suffix in ['png', 'pdf']:
        out_file = os.path.join(OUTPUT_DIR, f'combined_income_crown_area_scatter.{suffix}')
        fig.savefig(out_file, dpi=PUBLIC_DPI)
        print(f"Saved: {out_file}")
    plt.close(fig)


if __name__ == "__main__":
    plot_crown_area_histogram()
    plot_crown_area_income_scatter()
