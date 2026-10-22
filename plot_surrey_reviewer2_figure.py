#!/usr/bin/env python3
"""
plot_surrey_reviewer2_figure.py
Publication-grade 4-panel supplementary figure for Nature Communications Sustainability:
Surrey, BC Crown-Merging Sensitivity Analysis Addressing Reviewer 2.

High-readability edition with enlarged typography and clean panel presentation.
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Paths & Setup ─────────────────────────────────────────────────────────────
# Reads the scenario table written by run_surrey_reviewer2_sensitivity.py
BASE_DIR = os.environ.get("TREE_MAPPING_BASE_DIR", os.getcwd())
OUT_DIR = os.path.join(BASE_DIR, 'analysis_output', 'surrey_reviewer2_analysis')
CSV_PATH = os.path.join(OUT_DIR, 'surrey_sensitivity_scenarios_summary.csv')

# Style parameters with enlarged typography
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica Neue', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.size': 12.0,
    'axes.labelsize': 13.5,
    'axes.labelweight': 'bold',
    'axes.titlesize': 14.5,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 12.0,
    'ytick.labelsize': 12.0,
    'legend.fontsize': 11.5,
    'axes.linewidth': 1.0,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.major.size': 4.5,
    'ytick.major.size': 4.5,
    'lines.linewidth': 2.2
})

# Curated Okabe-Ito Color Palette
OI_BLUE       = '#0072B2'
OI_VERMILLION = '#D55E00'
OI_GREEN      = '#009E73'
OI_PURPLE     = '#8172B3'
OI_ORANGE     = '#E69F00'
OI_CHARCOAL   = '#2C3E50'
OI_GRAY       = '#7F7F7F'
OI_LIGHTGRAY  = '#E0E0E0'

def parse_ci(ci_str):
    """Extract float bounds from '[low, high]' string."""
    m = re.findall(r'[-+]?\d*\.\d+|\d+', ci_str)
    if len(m) >= 2:
        return float(m[0]), float(m[1])
    return 0.0, 0.0

def main():
    print("=" * 70)
    print("PLOTTING SURREY REVIEWER 2 SENSITIVITY PUBLICATION FIGURE (LARGE TEXT)")
    print("=" * 70)

    df = pd.read_csv(CSV_PATH)

    def get_row(kw):
        sub = df[df['Scenario'].str.contains(kw, case=False, regex=False)]
        if len(sub) == 0:
            raise ValueError(f"Scenario containing '{kw}' not found in CSV")
        return sub.iloc[0]

    row_base = df.iloc[0] # Original baseline

    # Figure geometry with generous spacing for large text
    fig, axes = plt.subplots(2, 2, figsize=(18.2, 11.2), dpi=300)
    plt.subplots_adjust(hspace=0.34, wspace=0.64, top=0.96, bottom=0.08, left=0.21, right=0.95)

    # Common X-axis range across all 4 panels
    X_MIN, X_MAX = -0.30, +0.12
    RHO_BASE = float(row_base['Spearman_Rho']) # -0.1432

    # ──────────────────────────────────────────────────────────────────────────
    # PANEL (a): Large-Component Exclusion Sensitivity
    # ──────────────────────────────────────────────────────────────────────────
    ax_a = axes[0, 0]
    scenarios_a = [
        ("Baseline (All Components)", row_base, "0.0% area excl."),
        ("Exclude > 200 m²", get_row("Exclude Components > 200"), "10.8% area excl."),
        ("Exclude > 100 m²", get_row("Exclude Components > 100"), "17.4% area excl."),
        ("Exclude > 50 m²",  get_row("Exclude Components > 50"),  "26.6% area excl."),
        ("Exclude > 25 m²",  get_row("Exclude Components > 25"),  "38.7% area excl.")
    ]
    
    y_pos_a = np.arange(len(scenarios_a))[::-1]
    labels_a = []
    
    for i, (label, row, note) in enumerate(scenarios_a):
        y = y_pos_a[i]
        rho = float(row['Spearman_Rho'])
        ci_low, ci_high = parse_ci(row['Rho_CI_95'])
        color = OI_CHARCOAL if i == 0 else OI_VERMILLION
        marker = 'D' if i == 0 else 'o'
        
        ax_a.errorbar(rho, y, xerr=[[rho - ci_low], [ci_high - rho]],
                      fmt=marker, color=color, ecolor=color, elinewidth=2.6,
                      capsize=5.5, capthick=2.0, markersize=9.5, zorder=4)
        
        ax_a.text(ci_high + 0.010, y, f"ρ = {rho:.3f}", va='center', ha='left',
                  fontsize=11.5, fontweight='bold', color=color)
        
        labels_a.append(f"{label}\n({note})")

    ax_a.set_yticks(y_pos_a)
    ax_a.set_yticklabels(labels_a, fontweight='medium', fontsize=11.5)
    ax_a.set_xlim(X_MIN, X_MAX)
    ax_a.set_xlabel("Spearman's ρ", fontweight='bold', fontsize=13.0)
    ax_a.set_title("(a) Large-Component Truncation Sensitivity", loc='left', pad=12, fontsize=14.0)

    # ──────────────────────────────────────────────────────────────────────────
    # PANEL (b): Small-Component Exclusion Sensitivity
    # ──────────────────────────────────────────────────────────────────────────
    ax_b = axes[0, 1]
    scenarios_b = [
        ("Baseline (All Components)", row_base, "0.0% comps excl."),
        ("Exclude < 3 px (~0.97 m²)", get_row("Exclude Components < 3 px"), "51.9% comps excl."),
        ("Exclude < 5 px (~1.62 m²)", get_row("Exclude Components < 5 px"), "65.0% comps excl."),
        ("Exclude < 10 px (~3.24 m²)", get_row("Exclude Components < 10 px"), "78.5% comps excl.")
    ]
    
    y_pos_b = np.arange(len(scenarios_b))[::-1]
    labels_b = []
    
    for i, (label, row, note) in enumerate(scenarios_b):
        y = y_pos_b[i]
        rho = float(row['Spearman_Rho'])
        ci_low, ci_high = parse_ci(row['Rho_CI_95'])
        color = OI_CHARCOAL if i == 0 else OI_GREEN
        marker = 'D' if i == 0 else 's'
        
        ax_b.errorbar(rho, y, xerr=[[rho - ci_low], [ci_high - rho]],
                      fmt=marker, color=color, ecolor=color, elinewidth=2.6,
                      capsize=5.5, capthick=2.0, markersize=9.5, zorder=4)
        
        ax_b.text(ci_high + 0.010, y, f"ρ = {rho:.3f}", va='center', ha='left',
                  fontsize=11.5, fontweight='bold', color=color)
        
        labels_b.append(f"{label}\n({note})")

    ax_b.set_yticks(y_pos_b)
    ax_b.set_yticklabels(labels_b, fontweight='medium', fontsize=11.5)
    ax_b.set_xlim(X_MIN, X_MAX)
    ax_b.set_xlabel("Spearman's ρ", fontweight='bold', fontsize=13.0)
    ax_b.set_title("(b) Small-Component Exclusion Sensitivity", loc='left', pad=12, fontsize=14.0)

    # ──────────────────────────────────────────────────────────────────────────
    # PANEL (c): Combined Component-Size Filtering
    # ──────────────────────────────────────────────────────────────────────────
    ax_c = axes[1, 0]
    scenarios_c = [
        ("Baseline (All Components)", row_base, "N = 570, 0.0% area excl."),
        ("Retain 3 px to 100 m²", get_row("Min 3 px (~0.97 m2) & Max 100 m2"), "N = 570, 23.4% area excl."),
        ("Retain 3 px to 50 m²",  get_row("Min 3 px (~0.97 m2) & Max 50 m2"),  "N = 570, 32.6% area excl."),
        ("Retain 10 px to 100 m²", get_row("Min 10 px (~3.24 m2) & Max 100 m2"), "N = 570, 35.4% area excl."),
        ("Retain 5 to 100 m²",    get_row("Retain 5 to 100 m2 Components"),    "N = 568, 43.0% area excl.")
    ]
    
    y_pos_c = np.arange(len(scenarios_c))[::-1]
    labels_c = []
    
    for i, (label, row, note) in enumerate(scenarios_c):
        y = y_pos_c[i]
        rho = float(row['Spearman_Rho'])
        ci_low, ci_high = parse_ci(row['Rho_CI_95'])
        color = OI_CHARCOAL if i == 0 else OI_PURPLE
        marker = 'D' if i == 0 else '^'
        
        ax_c.errorbar(rho, y, xerr=[[rho - ci_low], [ci_high - rho]],
                      fmt=marker, color=color, ecolor=color, elinewidth=2.6,
                      capsize=5.5, capthick=2.0, markersize=9.5, zorder=4)
        
        ax_c.text(ci_high + 0.010, y, f"ρ = {rho:.3f}", va='center', ha='left',
                  fontsize=11.5, fontweight='bold', color=color)
        
        labels_c.append(f"{label}\n({note})")

    ax_c.set_yticks(y_pos_c)
    ax_c.set_yticklabels(labels_c, fontweight='medium', fontsize=11.5)
    ax_c.set_xlim(X_MIN, X_MAX)
    ax_c.set_xlabel("Spearman's ρ", fontweight='bold', fontsize=13.0)
    ax_c.set_title("(c) Combined Band-Pass Filtering", loc='left', pad=12, fontsize=14.0)

    # ──────────────────────────────────────────────────────────────────────────
    # PANEL (d): Alternative Crown-Area Metrics
    # ──────────────────────────────────────────────────────────────────────────
    ax_d = axes[1, 1]
    scenarios_d = [
        ("Mean Crown Area (Baseline)", row_base, "Primary metric, N = 570"),
        ("75th Percentile Crown Area", get_row("75th Percentile"), "Upper established canopy"),
        ("90th Percentile Crown Area", get_row("90th Percentile"), "Mature canopy quantile"),
        ("Median Crown Area",          get_row("Median Crown Area"), "Quantized (82% tied to 2–3 px)")
    ]
    
    y_pos_d = np.arange(len(scenarios_d))[::-1]
    labels_d = []
    
    for i, (label, row, note) in enumerate(scenarios_d):
        y = y_pos_d[i]
        rho = float(row['Spearman_Rho'])
        ci_low, ci_high = parse_ci(row['Rho_CI_95'])
        color = OI_CHARCOAL if i == 0 else OI_ORANGE
        marker = 'D' if i == 0 else 'v'
        
        ax_d.errorbar(rho, y, xerr=[[rho - ci_low], [ci_high - rho]],
                      fmt=marker, color=color, ecolor=color, elinewidth=2.6,
                      capsize=5.5, capthick=2.0, markersize=9.5, zorder=4)
        
        ax_d.text(ci_high + 0.010, y, f"ρ = {rho:.3f}", va='center', ha='left',
                  fontsize=11.5, fontweight='bold', color=color)
        
        labels_d.append(f"{label}\n({note})")

    ax_d.set_yticks(y_pos_d)
    ax_d.set_yticklabels(labels_d, fontweight='medium', fontsize=11.5)
    ax_d.set_xlim(X_MIN, X_MAX)
    ax_d.set_xlabel("Spearman's ρ", fontweight='bold', fontsize=13.0)
    ax_d.set_title("(d) Alternative Crown-Area Metrics", loc='left', pad=12, fontsize=14.0)

    # Format all panels consistently
    for ax in [ax_a, ax_b, ax_c, ax_d]:
        ax.grid(True, linestyle='--', alpha=0.35, color='#B0BEC5', zorder=1)
        # Vertical reference line at baseline rho = -0.1432
        ax.axvline(RHO_BASE, color='#555555', linestyle='--', linewidth=1.3, alpha=0.75, zorder=2)
        # Vertical dotted zero line
        ax.axvline(0.0, color='#888888', linestyle=':', linewidth=1.1, alpha=0.70, zorder=2)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.025))

    # Save Outputs (without any bottom caption or top suptitle)
    png_path = os.path.join(OUT_DIR, 'surrey_reviewer2_sensitivity_figure.png')
    pdf_path = os.path.join(OUT_DIR, 'surrey_reviewer2_sensitivity_figure.pdf')
    svg_path = os.path.join(OUT_DIR, 'surrey_reviewer2_sensitivity_figure.svg')

    plt.savefig(png_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.savefig(pdf_path, facecolor='white', bbox_inches='tight')
    plt.savefig(svg_path, facecolor='white', bbox_inches='tight')
    plt.close()

    print(f"[OK] Figure exported successfully:")
    print(f"  PNG: {png_path}")
    print(f"  PDF: {pdf_path}")
    print(f"  SVG: {svg_path}")
    print("=" * 70)

if __name__ == '__main__':
    main()
