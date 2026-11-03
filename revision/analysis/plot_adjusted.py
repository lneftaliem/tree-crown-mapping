"""
plot_adjusted.py -- Extended Data Fig. 3 (spatial model sensitivity) and Fig. 4 (adjusted income coefficients).

The original plotting script was not found (only its outputs, sherlock_venv/analysis_output/equity_results/
figure_{spatial_model_sensitivity,adjusted_income_coefficients}_refined.{png,pdf,svg}); this re-creates that design:
three panels (tree density, canopy cover, mean crown area), cities top to bottom in manuscript order with the
manuscript city colours (Okabe-Ito, as in generate_plots_corrected.py), filled circle = HC3 OLS, open square = Queen
spatial error, open triangle = 5-NN spatial error, capped 95 % CI bars, dotted zero line, shared x label.
x-limits are symmetric about zero and set from the data (they are not fixed to the original limits, so differences
are not hidden). Reads analysis_output/adjusted_models/{ols_hc3,spatial_error}.csv.
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = os.environ['TREE_MAPPING_BASE_DIR']
IN = os.path.join(BASE, 'analysis_output', 'adjusted_models')
OUT = os.path.join(BASE, 'analysis_output', 'extended_data')
os.makedirs(OUT, exist_ok=True)
CITIES = ['austin', 'bloomington', 'cupertino', 'surrey']
LABEL = {'austin': 'Austin, TX', 'bloomington': 'Bloomington, IN', 'cupertino': 'Cupertino, CA', 'surrey': 'Surrey, BC'}
COL = {'austin': '#E69F00', 'bloomington': '#56B4E9', 'cupertino': '#009E73', 'surrey': '#CC79A7'}
PANELS = [('Tree Density', '(a) Tree density'), ('Canopy Cover', '(b) Canopy cover'), ('Crown Area', '(c) Mean crown area')]
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'pdf.fonttype': 42,
                     'svg.fonttype': 'none', 'axes.edgecolor': '#333333', 'axes.linewidth': 0.9})
ols = pd.read_csv(f'{IN}/ols_hc3.csv')
sp = pd.read_csv(f'{IN}/spatial_error.csv')


def nice_lim(v):
    v = max(v * 1.08, 1e-3)
    e = 10 ** np.floor(np.log10(v))
    return np.ceil(v / e * 2) / 2 * e


def draw(spatial, name):
    h = 3.9 if spatial else 3.1
    fig, axes = plt.subplots(1, 3, figsize=(10.0, h), sharey=True)
    for ax, (oc, title) in zip(axes, PANELS):
        lim = 0
        for i, c in enumerate(CITIES):
            y0 = len(CITIES) - 1 - i
            o = ols[(ols.city == c) & (ols.outcome == oc)].iloc[0]
            s = sp[(sp.city == c) & (sp.outcome == oc)].iloc[0]
            rows = [(o.beta, o.ci_low, o.ci_high, 'o', True, -0.24 if spatial else 0)]
            if spatial:
                rows += [(s.queen_beta, s.queen_ci_low, s.queen_ci_high, 's', False, 0),
                         (s.knn5_beta, s.knn5_ci_low, s.knn5_ci_high, '^', False, 0.24)]
            for b, lo, hi, mk, filled, dy in rows:
                ax.errorbar(b, y0 + dy, xerr=[[b - lo], [hi - b]], fmt=mk, color=COL[c], ms=7 if filled else 7,
                            mfc=COL[c] if filled else 'white', mec=COL[c], mew=1.6, elinewidth=1.6, capsize=3.5, capthick=1.6)
                lim = max(lim, abs(lo), abs(hi))
        L = nice_lim(lim)
        ax.set_xlim(-L, L)
        ax.axvline(0, color='#555555', ls=':', lw=1.2, zorder=0)
        ax.grid(axis='x', color='#cccccc', ls=':', lw=0.7)
        ax.set_title(title, loc='left', fontweight='bold', fontsize=11)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=9.5)
    axes[0].set_yticks(range(len(CITIES)))
    axes[0].set_yticklabels([LABEL[c] for c in reversed(CITIES)], fontsize=10)
    axes[0].set_ylim(-0.6, len(CITIES) - 0.4)
    fig.supxlabel('Adjusted income coefficient', fontsize=10.5)
    if spatial:
        hd = [Line2D([], [], marker='o', color='#555555', ls='', ms=7, label='HC3 OLS (Baseline)'),
              Line2D([], [], marker='s', color='#555555', mfc='white', mew=1.6, ls='', ms=7, label='Queen Spatial Error'),
              Line2D([], [], marker='^', color='#555555', mfc='white', mew=1.6, ls='', ms=7, label='KNN (k = 5) Spatial Error')]
        fig.legend(handles=hd, loc='upper center', ncol=3, frameon=False, fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
    else:
        fig.tight_layout()
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(f'{OUT}/{name}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)


draw(True, 'ED_Fig3_spatial_model_sensitivity')
draw(False, 'ED_Fig4_adjusted_income_coefficients')
print('written', sorted(os.listdir(OUT)))
