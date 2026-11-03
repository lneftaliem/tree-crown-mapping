"""
closed_blocks.py -- render random 'closed-canopy' 64 m blocks (>= 70 % of pixels NDVI >= 0.6, inside the city, tile
interior) per city with the kept crowns, to check visually what the low detection completeness in such blocks is.
Read-only. Usage: python closed_blocks.py <dataset> <boundaries_dir> <out_png> <n_per_city> <seed>
"""
import sys, os
import numpy as np, pandas as pd, geopandas as gpd, pyogrio
from shapely.geometry import Point
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chips
DS, BND, OUT, N, SEED = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
rng = np.random.default_rng(SEED)
fig, ax = plt.subplots(4, 2 * N, figsize=(3 * 2 * N, 13))
for ci, city in enumerate(chips.CITIES):
    k = pyogrio.read_dataframe(f'{DS}/{city}_ripley_points_with_income.geojson', read_geometry=False, columns=['source_tile', 'instance_label'])
    kept = k.groupby('source_tile').instance_label.apply(lambda x: set(x.astype(int)))
    muni = gpd.read_file(f'{BND}/{city}_exact_boundary.geojson').to_crs(4326).union_all()
    tiles = [f for f in chips.INV.index if chips.INV.loc[f, 'city'] == city]
    got = 0; tries = 0
    while got < N and tries < 3000:
        tries += 1
        fn = tiles[rng.integers(len(tiles))]; g = chips.GEO.loc[fn]; px = (g.R - g.L) / g.o_w
        r0, c0 = rng.integers(0, int(g.o_h) - 256), rng.integers(0, int(g.o_w) - 256)
        if not muni.contains(Point(g.L + (c0 + 128) * px, g['T'] - (r0 + 128) * px)):
            continue
        pan, nd, lab, _ = chips.window(fn, r0, c0, 256, 256)
        b = nd[64:192, 64:192]
        if not (np.isfinite(b).mean() > 0.95 and np.nanmean(b >= 0.6) >= 0.7):
            continue
        km = np.isin(lab, list(kept.get(fn, set()))) & (lab > 0)
        bd = find_boundaries(np.where(km, lab, 0), mode='inner') & km
        rgb = np.dstack([chips.stretch(pan)] * 3); rgb[bd] = (1, 0, 1)
        a0, a1 = ax[ci, 2 * got], ax[ci, 2 * got + 1]
        a0.imshow(rgb); a0.add_patch(plt.Rectangle((64, 64), 128, 128, fill=False, ec='cyan', lw=1))
        a0.set_title(f'{city[:3]} PAN; crowns cover {km[64:192,64:192].mean()*100:.0f}% of block', fontsize=8)
        a1.imshow(nd, cmap='RdYlGn', vmin=-0.1, vmax=0.8); a1.set_title('NDVI', fontsize=8)
        got += 1
    for a in ax[ci]: a.axis('off')
plt.tight_layout(); plt.savefig(OUT, dpi=70)
