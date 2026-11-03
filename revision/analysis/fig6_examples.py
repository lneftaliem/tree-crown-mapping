"""
fig6_examples.py -- Figure 6: example 256 x 256 pixel tiles per city -- (a) NDVI image, (b) binary prediction
(threshold 0.7; the semantic segmentation mask), (c) predicted tree crowns = the final instance-separated crowns
(boundaries of the separate_instances labels with projected area >= 3 m2, i.e. the crowns kept in the dataset, magenta)
over the NDVI image.

The original figure's tile windows were not recorded; windows are chosen by a fixed rule from the v1 tile inventory:
for each city, among selected in-city tiles, every non-overlapping 256-px window whose canopy fraction is 10-40 % is
scored by the number of instance-separated crowns >= 3 m2 that it contains, and the highest-scoring window is used
(ties: tile name, then position). NDVI is the source mosaic's band 2 ((NDVI + 1) * 1000) warped (nearest) onto the
prediction grid of the tile -- the inference grid. The model input is PAN (band 1, channel 0) + NDVI (band 2, channel 1;
v1b, reports/inference_channel_order.md), so panel (a) shows the NDVI channel the model saw (display warp: nearest neighbour onto the tile grid).
Dataset rule (2026-11-02): with a 4th argument (the final crown dataset), only crowns IN that dataset (source_tile,
instance_label) are counted for the window choice and drawn in panel (c). Without it, the rule counted crowns of the
unfiltered tile rasters, which selected open-water windows in Austin and Bloomington (crowns removed by the NDVI mask)
and a Surrey window outside the municipality.
Usage: python fig6_examples.py <tile_inventory.csv> <georef_table.csv> <out_dir> [<dataset_dir>]
"""
import os, sys
import numpy as np, pandas as pd, rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from affine import Affine
from skimage.segmentation import find_boundaries
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

inv, geo, OUT = pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2]).set_index('filename'), sys.argv[3]
DS = sys.argv[4] if len(sys.argv) > 4 else None
# optional (display only): skip windows with a crop-row texture signature (work_v1b/diag/row_texture blocks, max score of
# the window's four 64 m blocks > 100; see code/analysis/landcover/row_texture.py). The dataset is unchanged.
RT = sys.argv[5] if len(sys.argv) > 5 else None
os.makedirs(OUT, exist_ok=True)
PGC = '/path/to/Urban_data/NASA:PGC/Training_06062024'
CITIES = ['austin', 'bloomington', 'cupertino', 'surrey']
W = 256
sel = inv[inv.decision == 'v1_ndvi_pan']
assert len(sel) == len(inv) and set(sel.city) <= set(CITIES)


def pred_path(r):
    return r.pred_source


_rt = {}


def rt_max(fn, i, j):
    if fn not in _rt:
        b = pd.read_parquet(f'{RT}/blocks_{fn}.parquet'); _rt[fn] = b.set_index(['block_r', 'block_c']).score
    s = _rt[fn].reindex([(i // 128 + a, j // 128 + c) for a in (0, 1) for c in (0, 1)])
    return float(s.max()) if s.notna().any() else 0.0


best, kept = {}, {}
for city in CITIES:
    if DS:
        import pyogrio
        k = pyogrio.read_dataframe(f'{DS}/{city}_ripley_points_with_income.geojson', read_geometry=False,
                                   columns=['source_tile', 'instance_label'])
        kept[city] = {t: set(g.instance_label.astype('int64')) for t, g in k.groupby('source_tile')}
    cand = []
    for r in sel[sel.city == city].sort_values('filename').itertuples():
        with rasterio.open(r.split_source) as s:
            lab = s.read(1)
        with rasterio.open(pred_path(r)) as p:
            pr = p.read(1) > 0
        px = (geo.loc[r.filename, 'R'] - geo.loc[r.filename, 'L']) / lab.shape[1]
        a = (px * 111320 * np.cos(np.radians(geo.loc[r.filename, 'T']))) * (px * 110574)
        sizes = np.bincount(lab.ravel())
        for i in range(0, lab.shape[0] - W + 1, W):
            for j in range(0, lab.shape[1] - W + 1, W):
                f = pr[i:i + W, j:j + W].mean()
                if not 0.10 <= f <= 0.40:
                    continue
                if RT is not None and rt_max(r.filename, i, j) > 100:
                    continue
                u = np.unique(lab[i:i + W, j:j + W]); u = u[u > 0]
                if DS:
                    u = u[np.isin(u, list(kept[city].get(r.filename, ())))]
                n = int((sizes[u] * a >= 3).sum())
                cand.append((n, r.filename, i, j))
    cand.sort(key=lambda t: (-t[0], t[1], t[2], t[3]))
    best[city] = cand[0]
    print(city, cand[0], flush=True)

fig, axes = plt.subplots(4, 3, figsize=(10.5, 14.2))
for k, city in enumerate(CITIES):
    n, fn, i, j = best[city]
    r = sel[sel.filename == fn].iloc[0]
    g = geo.loc[fn]
    px = (g.R - g.L) / g.o_w
    tr = Affine(px, 0, g.L + j * px, 0, -px, g['T'] - i * px)
    sid = '_'.join(r.mosaic.split('_')[-3:-1])
    mosaic = f'{PGC}/{city}/{sid}/{r.mosaic}'
    with rasterio.open(mosaic) as m, WarpedVRT(m, crs='EPSG:4326', transform=tr, width=W, height=W, resampling=Resampling.nearest) as v:
        raw = v.read(2).astype(float)
    ndvi = np.where(raw > 0, raw / 1000 - 1, np.nan)
    with rasterio.open(pred_path(r)) as p:
        b = p.read(1, window=((i, i + W), (j, j + W))) > 0
    with rasterio.open(r.split_source) as s:
        lab = s.read(1, window=((i, i + W), (j, j + W)))
    a_px = (px * 111320 * np.cos(np.radians(g['T']))) * (px * 110574)
    with rasterio.open(r.split_source) as s:
        full = np.bincount(s.read(1).ravel())
    keep = full * a_px >= 3; keep[0] = False
    if DS:
        inds = np.zeros(len(keep), bool); ids = [x for x in kept[city].get(fn, ()) if x < len(keep)]; inds[ids] = True
        keep &= inds
    lab = np.where(keep[lab], lab, 0)
    bnd = find_boundaries(lab, mode='inner') & (lab > 0)
    ax = axes[k]
    ax[0].imshow(ndvi, cmap='RdYlGn', vmin=-0.1, vmax=0.8, interpolation='nearest')
    ax[1].imshow(b, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    ax[2].imshow(ndvi, cmap='RdYlGn', vmin=-0.1, vmax=0.8, alpha=0.6, interpolation='nearest')
    over = np.zeros((W, W, 4)); over[bnd] = (1, 0, 1, 1)
    ax[2].imshow(over, interpolation='nearest')
    bar = 25 / (px * 111320 * np.cos(np.radians(g['T'])))
    for a_ in ax:
        a_.set_xticks([]); a_.set_yticks([])
        a_.plot([10, 10 + bar], [W - 10, W - 10], color='black', lw=2.5)
        a_.text(10 + bar / 2, W - 16, '25 m', ha='center', va='bottom', fontsize=9.5, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='black', lw=0.6))
    ax[0].set_ylabel(city.title(), fontsize=14, fontweight='bold')
    pd.DataFrame([dict(city=city, tile=fn, row=i, col=j, crowns_ge3m2=n, binary_fraction=float(b.mean()),
                       n_binary_components=int(__import__('scipy.ndimage', fromlist=['label']).label(b)[1]),
                       n_instances=int(len(np.unique(lab)) - 1))]).to_csv(f'{OUT}/Fig6_windows.csv', mode='a',
                                                                            header=k == 0, index=False)
for a_, t in zip(axes[0], ['(a) NDVI Image', '(b) Binary prediction (τ = 0.7)', '(c) Separated crowns in dataset']):
    a_.set_title(t, fontsize=11.5, fontweight='bold')
fig.suptitle('Example 256×256 Pixel Tiles by City', fontsize=15, fontweight='bold')
fig.tight_layout(rect=(0, 0, 1, 0.975))
for ext in ('png', 'pdf'):
    fig.savefig(f'{OUT}/Fig6_example_tiles.{ext}', dpi=300, bbox_inches='tight')
print('written', OUT)
