#!/usr/bin/env python3
"""
generate_plots_corrected.py

Cross-city income vs. tree metric scatter plots using the latitude-corrected
crown areas (see run_crown_area_correction_pipeline.py):
  - combined_income_zscore_density_scatter.{png,pdf,svg}
  - combined_income_zscore_canopy_cover_scatter_corrected.{png,pdf,svg}
  - combined_income_zscore_crown_area_scatter_corrected.{png,pdf,svg}

Input: data/covariates/all_cities_urban_covariates_corrected.csv (committed;
staged automatically into $TREE_MAPPING_BASE_DIR/analysis_output/equity_results/).
Outputs are written to $TREE_MAPPING_BASE_DIR/analysis_output/equity_results/.
"""
import os
import sys
import math
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use('Agg')

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
import repro_data  # noqa: E402

BASE_DIR = repro_data.BASE_DIR
EQUITY_OUT_DIR = repro_data.EQUITY_STAGE_DIR

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.gridspec as gridspec  # noqa: E402
import seaborn as sns  # noqa: E402

CITIES = ['austin', 'bloomington', 'cupertino', 'surrey']

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

def fmt_sig(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return ' ns'

def sigfig_rho_2(val):
    """Format correlation rho to 2 significant figures."""
    if val == 0: return '0'
    abs_v = abs(val)
    power = math.floor(math.log10(abs_v))
    decimals = max(2, 1 - power)
    factor = 10 ** (1 - power)
    rounded = round(abs_v * factor) / factor
    sign = '-' if val < 0 else ''
    return f"{sign}{rounded:.{decimals}f}"

def sigfig_p_2(p):
    """Format p-value to 2 significant figures."""
    if p == 0: return '0'
    if p < 0.001:
        power = math.floor(math.log10(p))
        mantissa = round(p / (10**power), 1)
        if mantissa >= 10.0:
            mantissa /= 10.0
            power += 1
        return f"{mantissa:.1f}e{power}"
    abs_v = abs(p)
    power = math.floor(math.log10(abs_v))
    decimals = max(2, 1 - power)
    factor = 10 ** (1 - power)
    rounded = round(abs_v * factor) / factor
    return f"{rounded:.{decimals}f}"

def generate_scatter_plot(df_dict, x_col, y_col, y_label, y_ticks, y_ticklabels,
                          y_max_plot, output_basename, clip_y=None, leg_loc='upper right',
                          anchor=(0.99, 0.99), is_percent=False):
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

    all_x = []
    for city in CITIES:
        sub = df_dict[city].dropna(subset=[x_col, y_col])
        all_x.extend(sub[x_col].values.tolist())
    x_min, x_max = min(all_x), max(all_x)
    x_lim = (np.floor(x_min - 0.3), np.ceil(x_max + 0.3))

    np.random.seed(42)

    for city in CITIES:
        sub = df_dict[city].dropna(subset=[x_col, y_col])
        color = OI_COLORS[city]
        label = CITY_LABELS[city]

        x = sub[x_col].values
        y = sub[y_col].values

        rho, p_val = stats.spearmanr(x, y)
        stars = fmt_sig(p_val)
        
        # 2 SIGNIFICANT FIGURES FOR RHO AND P-VALUE:
        rho_str = sigfig_rho_2(rho)
        p_str = sigfig_p_2(p_val)
        leg_str = f"{label} $\\rho = {rho_str}${stars} ($p = {p_str}, n = {len(sub)}$)"

        y_val_plot = np.clip(y, 0, clip_y) if clip_y is not None else y
        y_trans = np.log10(y_val_plot + 1.0)

        # Uniform jitter for exact zeros
        zero_mask = (y_val_plot == 0)
        if np.any(zero_mask):
            y_trans[zero_mask] += np.random.uniform(-0.015, 0.015, size=np.sum(zero_mask))

        path = ax_main.scatter(x, y_trans, s=10, alpha=0.55, color=color,
                               edgecolors='white', linewidths=0.2, zorder=20)
        legend_handles.append(path)
        legend_labels.append(leg_str)

        # Marginals
        sns.kdeplot(x=x, ax=ax_top, color=color, fill=True, alpha=0.25, lw=0, bw_adjust=1.0)
        sns.kdeplot(x=x, ax=ax_top, color=color, fill=False, alpha=0.9, lw=1.0, bw_adjust=1.0)
        sns.kdeplot(y=y_trans, ax=ax_right, color=color, fill=True, alpha=0.25, lw=0, bw_adjust=1.0)
        sns.kdeplot(y=y_trans, ax=ax_right, color=color, fill=False, alpha=0.9, lw=1.0, bw_adjust=1.0)

        # OLS in log10 space on UNCLIPPED y
        try:
            x_eval = np.linspace(x.min(), x.max(), 100)
            slope, intercept = np.polyfit(x, np.log10(y + 1.0), 1)
            z_eval = intercept + slope * x_eval
            ax_main.plot(x_eval, z_eval, color=color, lw=2.0, alpha=1.0, zorder=15)

            boots = []
            for _ in range(1000):
                idx = np.random.choice(len(x), len(x), replace=True)
                xb, yb = x[idx], y[idx]
                try:
                    sb, ib = np.polyfit(xb, np.log10(yb + 1.0), 1)
                    boots.append(ib + sb * x_eval)
                except: pass
            if boots:
                boots = np.array(boots)
                lower = np.percentile(boots, 2.5, axis=0)
                upper = np.percentile(boots, 97.5, axis=0)
                ax_main.fill_between(x_eval, lower, upper, color=color, alpha=0.20, lw=0, zorder=14)
        except Exception as e:
            print(f"OLS failed for {city}: {e}")

    for sax in [ax_top, ax_right]:
        sax.axis('off')

    ax_main.set_xlim(x_lim[0], x_lim[1])
    ax_main.set_ylim(0, np.log10(y_max_plot + 1.0))
    ax_main.set_yticks(np.log10(np.array(y_ticks) + 1.0))
    ax_main.set_yticklabels(y_ticklabels)

    ax_main.set_xlabel("Relative Median Household Income (within-city z-score)", fontsize=10, fontweight='bold')
    ax_main.set_ylabel(y_label, fontsize=10, fontweight='bold')

    ax_main.tick_params(axis='both', which='major', labelsize=8.5)
    ax_main.grid(False)
    ax_main.spines['right'].set_visible(False)
    ax_main.spines['top'].set_visible(False)

    leg = ax_main.legend(legend_handles, legend_labels, loc=leg_loc,
                         bbox_to_anchor=anchor, frameon=True, fontsize=7.6,
                         borderpad=0.25, labelspacing=0.25)
    leg.get_frame().set_alpha(0.7)
    leg.get_frame().set_edgecolor('none')
    for h in leg.legend_handles:
        h.set_sizes([30.0])
        h.set_alpha(0.9)

    plt.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.12)

    for out_d in [EQUITY_OUT_DIR]:
        os.makedirs(out_d, exist_ok=True)
        for ext in ['png', 'pdf', 'svg']:
            out_path = os.path.join(out_d, f"{output_basename}.{ext}")
            fig.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {out_path}")
    plt.close(fig)

def main():
    repro_data.stage_covariates()
    master_path = os.path.join(EQUITY_OUT_DIR, "all_cities_urban_covariates_corrected.csv")
    df = pd.read_csv(master_path)

    canopy_dict = {c: df[df['city'] == c] for c in CITIES}
    crown_dict = {c: df[(df['city'] == c) & (df['tree_count'] >= 10)] for c in CITIES}

    print("Generating Figure 1: Tree Density vs Income Z-score (2 sig figs)...")
    generate_scatter_plot(
        df_dict=canopy_dict,
        x_col='income_zscore',
        y_col='tree_density_per_km2',
        y_label=r"Tree Density (trees km$^{\mathbf{-2}}$, log$_{\mathbf{10}}$(y+1) scale)",
        y_ticks=[0, 10, 100, 1000, 10000, 30000],
        y_ticklabels=['0', '10', '100', '1K', '10K', '30K'],
        y_max_plot=33000,
        output_basename="combined_income_zscore_density_scatter",
        leg_loc='lower left',
        anchor=(0.02, 0.02)
    )

    print("Regenerating Corrected Figure: Canopy Cover vs Income Z-Score (2 sig figs)...")
    generate_scatter_plot(
        df_dict=canopy_dict,
        x_col='income_zscore',
        y_col='canopy_cover_pct_corrected',
        y_label=r"Canopy Cover (%, log$_{\mathbf{10}}$(y+1) scale)",
        y_ticks=[0, 0.5, 1, 2, 5, 10, 25, 50, 100],
        y_ticklabels=['0%', '0.5%', '1%', '2%', '5%', '10%', '25%', '50%', '100%'],
        y_max_plot=100,
        output_basename="combined_income_zscore_canopy_cover_scatter_corrected",
        leg_loc='upper right',
        anchor=(0.99, 0.99),
        is_percent=True
    )

    print("Regenerating Corrected Figure: Mean Crown Area vs Income Z-Score (2 sig figs)...")
    generate_scatter_plot(
        df_dict=crown_dict,
        x_col='income_zscore',
        y_col='mean_crown_area_m2',
        y_label=r"Mean Crown Area per Tree (m$^{\mathbf{2}}$, log$_{\mathbf{10}}$(y+1) scale)",
        y_ticks=[0, 1, 2, 5, 10, 20, 50, 100],
        y_ticklabels=['0', '1', '2', '5', '10', '20', '50', '100'],
        y_max_plot=100,
        clip_y=100.0,
        output_basename="combined_income_zscore_crown_area_scatter_corrected",
        leg_loc='upper right',
        anchor=(0.99, 0.99)
    )

if __name__ == '__main__':
    main()
