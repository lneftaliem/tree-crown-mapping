"""
fig6_selected.py -- Figure 6 from user-selected windows (presentation only; read-only on all data; PAN never read).
Same three-column layout as fig6_examples.py: (a) NDVI image, (b) binary prediction (tau = 0.7), (c) separated crowns of
the final dataset (tree_crowns_v1b_ndvi02) over NDVI; one row per city, 25 m scale bars. Panel drawing is shared with
contact_sheets.py. Windows (2026-11-02, chosen by the authors from the NDVI-only candidate sheets in
outputs/fig6_review): Austin A3, Bloomington B3, Cupertino unchanged, Surrey S7.
Usage: python fig6_selected.py <selected_windows.csv> <out_dir>
"""
import sys, os
import numpy as np, pandas as pd, pyogrio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import contact_sheets as cs
import matplotlib.pyplot as plt

S, OUT = pd.read_csv(sys.argv[1]), sys.argv[2]
assert list(S.city) == list(cs.CITIES)
assert not os.path.exists(f'{OUT}/Fig6_example_tiles.png'), 'refusing to overwrite an existing Figure 6'
os.makedirs(OUT, exist_ok=True)
fig, axes = plt.subplots(4, 3, figsize=(10.5, 14.2))
rows = []
for ax, w in zip(axes, S.itertuples()):
    k = pyogrio.read_dataframe(f'{cs.P}/dataset/tree_crowns_v1b_ndvi02/{w.city}_ripley_points_with_income.geojson',
                               read_geometry=False, columns=['source_tile', 'instance_label'])
    kept = {t: set(g.instance_label.astype('int64')) for t, g in k.groupby('source_tile')}
    ndvi, b, bnd, lab, mx = cs.panels(w.city, w.tile, w.row, w.col, kept)
    cs.draw(ax, ndvi, b, bnd, mx, label_fs=9.5)
    ax[0].set_ylabel(w.city.title(), fontsize=14, fontweight='bold')
    rows.append(dict(city=w.city, candidate=w.id, tile=w.tile, row=w.row, col=w.col,
                     crowns_in_dataset=int(len(np.unique(lab)) - 1), binary_fraction=float(b.mean())))
for a_, t in zip(axes[0], ['(a) NDVI image', '(b) Binary prediction (τ = 0.7)', '(c) Separated crowns in the final dataset']):
    a_.set_title(t, fontsize=11.5, fontweight='bold')
fig.suptitle('Example 256×256 Pixel Tiles by City', fontsize=15, fontweight='bold')
fig.tight_layout(rect=(0, 0, 1, 0.975))
for ext in ('png', 'pdf'):
    fig.savefig(f'{OUT}/Fig6_example_tiles.{ext}', dpi=300, bbox_inches='tight')
pd.DataFrame(rows).to_csv(f'{OUT}/Fig6_windows.csv', index=False)
print('written', OUT)
