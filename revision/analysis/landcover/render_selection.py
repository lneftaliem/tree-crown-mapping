"""
render_selection.py -- render a stratified selection of crowns (PAN + NDVI chips, 128 px, crown in magenta, other kept
crowns in yellow) for visual validation of the row-texture score. Read-only on data.
Usage: python render_selection.py <dataset> <selection.csv: chip,city,source_tile,instance_label,crown_area_m2,score> <out_prefix>
"""
import sys, os
import numpy as np, pandas as pd, pyogrio
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chips

DS, SEL, OUT = sys.argv[1:4]
s = pd.read_csv(SEL)
H = 64
kept = {}
for c in s.city.unique():
    k = pyogrio.read_dataframe(f'{DS}/{c}_ripley_points_with_income.geojson', read_geometry=False, columns=['source_tile', 'instance_label'])
    kept[c] = k.groupby('source_tile').instance_label.apply(lambda x: set(x.astype(int)))
rows = []
for r in s.itertuples():
    with __import__('rasterio').open(chips.INV.loc[r.source_tile, 'split_source']) as src:
        lab = src.read(1)
    rr, cc = np.argwhere(lab == r.instance_label).mean(0).astype(int)
    pan, nd, lb, _ = chips.window(r.source_tile, rr - H, cc - H, 2 * H, 2 * H)
    rows.append((pan, nd, lb, int(r.instance_label), kept[r.city].get(r.source_tile, set())))
for k in range(0, len(s), 16):
    fig, ax = plt.subplots(4, 8, figsize=(24, 12.6))
    for j, a2 in enumerate(ax.reshape(-1, 2)):
        for a in a2: a.axis('off')
        if k + j >= len(s): continue
        pan, nd, lb, L, kp = rows[k + j]
        tgt = find_boundaries(lb == L, mode='inner') & (lb == L)
        oth = np.isin(lb, list(kp)) & (lb != L) & (lb > 0)
        ob = find_boundaries(np.where(oth, lb, 0), mode='inner') & oth
        rgb = np.dstack([chips.stretch(pan)] * 3); rgb[ob] = (1, 0.85, 0); rgb[tgt] = (1, 0, 1)
        a2[0].imshow(rgb); a2[0].set_title(f'{s.chip[k+j]}  score {s.score[k+j]:.0f}  {s.crown_area_m2[k+j]:.0f} m²', fontsize=9)
        a2[1].imshow(nd, cmap='RdYlGn', vmin=-0.1, vmax=0.8); a2[1].set_title('NDVI', fontsize=9)
    plt.tight_layout(); plt.savefig(f'{OUT}_{k//16:02d}.png', dpi=72); plt.close()
print('done')
