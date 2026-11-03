#!/usr/bin/env python3
"""
plot_surrey_s12_three_panel.py -- Supplementary Fig. S12 (Surrey income-crown area association; crown-size sensitivity),
three-panel layout for the corrected (instance-separated, >= 3 m2) pipeline. Replaces plot_surrey_reviewer2_figure.py
(kept unchanged for the record), whose four-panel layout carried the original dataset's annotations as literals.

Reads only the scenario table written by run_surrey_reviewer2_sensitivity.py for THIS run; every number, label note and
axis limit is computed from it.
  (a) Large-crown exclusion       baseline; exclude crowns > 200, 100, 50, 25 m2
  (b) Combined crown-size filter  baseline; retain 5-100 m2. The 3 px-100 m2, 10 px-100 m2 and 3 px-50 m2 bands are
                                  omitted: with the pipeline's 3 m2 minimum they remove exactly the crowns removed by the
                                  > 100 m2 / > 50 m2 exclusions in (a) (identical rho, asserted below).
  (c) Alternative crown-area metrics  mean (baseline), 75th / 90th percentile, median per DA
The small-component exclusions (< 3, 5, 10 px) are not shown: all crowns are >= 3 m2, so they remove nothing
(identical to the baseline, asserted below).
"""
import os, re
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

BASE_DIR = os.environ.get("TREE_MAPPING_BASE_DIR", os.getcwd())
OUT_DIR = os.path.join(BASE_DIR, 'analysis_output', 'surrey_reviewer2_analysis')
df = pd.read_csv(os.path.join(OUT_DIR, 'surrey_sensitivity_scenarios_summary.csv'))

plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica Neue', 'Helvetica', 'DejaVu Sans'],
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 8, 'axes.linewidth': 0.8,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8})
CHAR, VERM, BLUE, ORNG = '#2C3E50', '#D55E00', '#0072B2', '#E69F00'


def row(kw):
    s = df[df.Scenario.str.contains(kw, case=False, regex=False)]
    assert len(s) == 1, kw
    return s.iloc[0]


def ci(r):
    lo, hi = (float(x) for x in re.findall(r'[-+]?\d*\.\d+|\d+', r.Rho_CI_95)[:2])
    return lo, hi


base = df.iloc[0]
same = lambda a, b: abs(float(a.Spearman_Rho) - float(b.Spearman_Rho)) < 1e-12 and abs(a.Excluded_Area_Pct - b.Excluded_Area_Pct) < 1e-9
for kw in ('< 3 px', '< 5 px', '< 10 px'):
    assert same(row(kw), base), f'small-component exclusion {kw} is not identical to the baseline'
assert same(row('Min 3 px (~0.97 m2) & Max 100'), row('Exclude Components > 100')) and same(row('Min 10 px'), row('Exclude Components > 100'))
assert same(row('Min 3 px (~0.97 m2) & Max 50'), row('Exclude Components > 50'))
N = int(base.Retained_DAs)

area = lambda r: f"{r.Excluded_Area_Pct:.1f}% of crown area removed"
panels = [
    ('(a) Large-crown exclusion', [
        ('All crowns (baseline)', base, f'mean crown area, n = {N} DAs', CHAR, 'D'),
        ('Exclude crowns > 200 m²', row('Exclude Components > 200'), area(row('Exclude Components > 200')), VERM, 'o'),
        ('Exclude crowns > 100 m²', row('Exclude Components > 100'), area(row('Exclude Components > 100')), VERM, 'o'),
        ('Exclude crowns > 50 m²', row('Exclude Components > 50'), area(row('Exclude Components > 50')), VERM, 'o'),
        ('Exclude crowns > 25 m²', row('Exclude Components > 25'), area(row('Exclude Components > 25')), VERM, 'o')]),
    ('(b) Combined crown-size filtering', [
        ('All crowns (baseline)', base, f'mean crown area, n = {N} DAs', CHAR, 'D'),
        ('Retain crowns 5–100 m²', row('Retain 5 to 100'), area(row('Retain 5 to 100')), BLUE, 's')]),
    ('(c) Alternative crown-area metrics', [
        ('Mean crown area (baseline)', base, 'primary metric', CHAR, 'D'),
        ('75th percentile crown area', row('75th Percentile'), 'per-DA quantile', ORNG, 'v'),
        ('90th percentile crown area', row('90th Percentile'), 'per-DA quantile', ORNG, 'v'),
        ('Median crown area', row('Median Crown Area'), 'per-DA quantile', ORNG, 'v')])]

los = [ci(r)[0] for _, rows in panels for _, r, *_ in rows]; his = [ci(r)[1] for _, rows in panels for _, r, *_ in rows]
x_min = np.floor((min(los) - 0.03) * 20) / 20
x_max = np.ceil((max(his) + 0.13) * 20) / 20  # room for the rho annotation right of each interval
rho_base = float(base.Spearman_Rho)

fig, axes = plt.subplots(3, 1, figsize=(7.0, 6.2), sharex=True, gridspec_kw=dict(height_ratios=[5, 2, 4], hspace=0.42))
fig.subplots_adjust(left=0.36, right=0.98, top=0.96, bottom=0.08)
for ax, (title, rows) in zip(axes, panels):
    y = np.arange(len(rows))[::-1]
    for yi, (lab, r, note, col, mk) in zip(y, rows):
        rho = float(r.Spearman_Rho); lo, hi = ci(r)
        ax.errorbar(rho, yi, xerr=[[rho - lo], [hi - rho]], fmt=mk, color=col, ecolor=col, elinewidth=1.4, capsize=3,
                    capthick=1.1, markersize=5.5, zorder=4)
        ax.text(hi + 0.012, yi, f'ρ = {rho:.3f}'.replace('-', '\u2212'), va='center', ha='left', fontsize=7.5, fontweight='bold', color=col)
    ax.set_yticks(y)
    ax.set_yticklabels([f'{lab}\n({note})' for lab, _, note, _, _ in rows], fontsize=7.5, linespacing=1.15)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.axvline(rho_base, color='#555555', linestyle='--', linewidth=0.9, alpha=0.8, zorder=2)
    ax.grid(axis='x', linestyle='--', alpha=0.35, color='#B0BEC5', zorder=1)
    ax.set_title(title, loc='left', fontsize=9, fontweight='bold', pad=4)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
axes[-1].set_xlim(x_min, x_max)
axes[-1].xaxis.set_major_locator(ticker.MultipleLocator(0.05))
axes[-1].set_xlabel("Spearman's ρ (DA median household income vs crown-area metric; 95% CI)", fontsize=8, fontweight='bold')
for ext in ('png', 'pdf', 'svg'):
    fig.savefig(os.path.join(OUT_DIR, f'surrey_reviewer2_sensitivity_figure.{ext}'), dpi=300, facecolor='white', bbox_inches='tight')
print('x-axis', x_min, x_max, '; written', OUT_DIR)
