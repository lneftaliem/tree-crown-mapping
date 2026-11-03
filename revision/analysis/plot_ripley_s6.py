"""
plot_ripley_s6.py -- Supplementary Fig. S6 from ripley_L_by_quartile.csv of a run (plot only; statistics unchanged,
computed by ripley_l.py). Area-weighted quartile curves of L(r), r = 2-50 m; y-limits from the data (negative values at
short range are shown, not clipped); the r <= 4 m range, where instance separation makes L(r) < 0 by construction
(crown centres >= ~4 m apart), is shaded and labelled "separation limit (r ≤ 4 m)" inside the band.
Usage: TREE_MAPPING_BASE_DIR=<run> python plot_ripley_s6.py
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
B = os.environ['TREE_MAPPING_BASE_DIR']; OUT = f'{B}/analysis_output/ripley'
q = pd.read_csv(f'{OUT}/ripley_L_by_quartile.csv')
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'pdf.fonttype': 42})
Lc = [c for c in q.columns if c.startswith('L_')]; R = np.array([int(c[2:]) for c in Lc])
lo, hi = q[Lc].values.min(), q[Lc].values.max(); pad = 0.06 * (hi - lo)
COL = {1: '#0072B2', 2: '#E69F00', 3: '#009E73', 4: '#D55E00'}
LAB = {1: 'Q1 (lowest income)', 2: 'Q2', 3: 'Q3', 4: 'Q4 (highest income)'}
fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.2), sharex=True, sharey=True)
for ax, city in zip(axes.flat, ['austin', 'bloomington', 'cupertino', 'surrey']):
    ax.axvspan(0, 5, color='#e9ecef', zorder=0)
    ax.text(2.5, 1.0, 'separation limit (r ≤ 4 m)', rotation=90, ha='center', va='bottom', fontsize=6,
            style='italic', color='#495057', zorder=2)
    ax.axhline(0, color='#555555', lw=0.8, zorder=1)
    for _, r in q[q.city == city].sort_values('quartile').iterrows():
        ax.plot(R, r[Lc].values.astype(float), color=COL[int(r.quartile)], lw=1.6, label=f'{LAB[int(r.quartile)]} (n = {int(r.n)})', zorder=3)
    ax.set_title(city.title(), fontsize=10, fontweight='bold')
    ax.grid(True, color='#dee2e6', lw=0.5); ax.tick_params(labelsize=8)
    ax.set_xlim(0, 51); ax.set_ylim(lo - pad, hi + pad)
    ax.legend(fontsize=6.5, loc='upper left', frameon=False, handlelength=1.4)
for ax in axes[1]:
    ax.set_xlabel('Distance $r$ (m)', fontsize=9, fontweight='bold')
for ax in axes[:, 0]:
    ax.set_ylabel(r'$\widehat{L}(r)$ (m)', fontsize=9, fontweight='bold')
# shaded band (r <= 4 m): labelled in the panels; full explanation in the caption
fig.tight_layout()
for ext in ('png', 'pdf', 'svg'):
    fig.savefig(f'{OUT}/FigS6_ripley_L_by_income_quartile.{ext}', dpi=300, bbox_inches='tight')
print('written', OUT)
