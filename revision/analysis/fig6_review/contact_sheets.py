"""
contact_sheets.py -- NDVI-only Figure 6 candidate contact sheets (presentation only; read-only on all data).
PAN is never read here: only mosaic band 2 (NDVI), the binary prediction and the instance labels. For each candidate
window in candidates.csv (id, city, tile, row, col, description), one row of three panels in the Figure 6 style:
NDVI | binary prediction (tau = 0.7) | separated crowns of the final dataset (tree_crowns_v1b_ndvi02) over NDVI.
Window metrics come from screening/scored_<city>.csv where the window was screened.
Writes <out>/Fig6_candidates_<city>.{png,pdf} and <out>/Fig6_candidates_metrics.csv.
Usage: python contact_sheets.py <candidates.csv> <out_dir>
"""
import sys, os
import numpy as np, pandas as pd, rasterio, pyogrio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from affine import Affine
from skimage.segmentation import find_boundaries
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import os
P = os.environ.get('URBAN_DATA', '/path/to/Urban_data') + '/tree_crowns_v1_20261101'
PGC = os.environ.get('URBAN_DATA', '/path/to/Urban_data') + '/NASA:PGC/Training_06062024'
CITIES = ('austin', 'bloomington', 'cupertino', 'surrey')
C, OUT = pd.read_csv(sys.argv[1]), sys.argv[2]
assert set(C.city) <= set(CITIES)
INV = pd.read_csv(f'{P}/work_v1b/tiles/tile_inventory.csv').set_index('filename')
GEO = pd.read_csv(f'{P}/work_v1b/diag/georef_table.csv').set_index('filename')
W = 256
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica Neue', 'DejaVu Sans']})


def panels(city, fn, i, j, kept):
    """NDVI (band 2 only), binary prediction, dataset-crown boundaries, metres per pixel (x)."""
    r, g = INV.loc[fn], GEO.loc[fn]
    assert r.city == city
    px = (g.R - g.L) / g.o_w
    tr = Affine(px, 0, g.L + j * px, 0, -px, g['T'] - i * px)
    sid = '_'.join(r.mosaic.split('_')[-3:-1])
    with rasterio.open(f'{PGC}/{city}/{sid}/{r.mosaic}') as m, \
            WarpedVRT(m, crs='EPSG:4326', transform=tr, width=W, height=W, resampling=Resampling.nearest) as v:
        raw = v.read(2).astype(float)
    ndvi = np.where(raw > 0, raw / 1000 - 1, np.nan)
    with rasterio.open(r.pred_source) as p:
        b = p.read(1, window=((i, i + W), (j, j + W))) > 0
    with rasterio.open(r.split_source) as s:
        lab = s.read(1, window=((i, i + W), (j, j + W)))
    lab = np.where(np.isin(lab, list(kept.get(fn, ()))), lab, 0)
    bnd = find_boundaries(lab, mode='inner') & (lab > 0)
    return ndvi, b, bnd, lab, px * 111320 * np.cos(np.radians(g['T']))


def draw(ax, ndvi, b, bnd, mx, label_fs=8):
    ax[0].imshow(ndvi, cmap='RdYlGn', vmin=-0.1, vmax=0.8, interpolation='nearest')
    ax[1].imshow(b, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    ax[2].imshow(ndvi, cmap='RdYlGn', vmin=-0.1, vmax=0.8, alpha=0.6, interpolation='nearest')
    o = np.zeros((W, W, 4)); o[bnd] = (1, 0, 1, 1)
    ax[2].imshow(o, interpolation='nearest')
    bar = 25 / mx
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
        # bar and label share one white box, so the bar reads the same on the NDVI and the black binary panels
        a.add_patch(FancyBboxPatch((6, W - 37), bar + 8, 31, boxstyle='round,pad=0,rounding_size=2', fc='white',
                                   ec='black', lw=0.6, zorder=5))
        a.plot([10, 10 + bar], [W - 11, W - 11], color='black', lw=2.5, solid_capstyle='butt', zorder=6)
        a.text(10 + bar / 2, W - 16, '25 m', ha='center', va='bottom', fontsize=label_fs, fontweight='bold', zorder=6)
        a.set_xlim(-0.5, W - 0.5); a.set_ylim(W - 0.5, -0.5)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for city, cc in C.groupby('city', sort=False):
        k = pyogrio.read_dataframe(f'{P}/dataset/tree_crowns_v1b_ndvi02/{city}_ripley_points_with_income.geojson',
                                   read_geometry=False, columns=['source_tile', 'instance_label'])
        kept = {t: set(g.instance_label.astype('int64')) for t, g in k.groupby('source_tile')}
        sc = f'{OUT}/screening/scored_{city}.csv'
        sc = pd.read_csv(sc).set_index(['tile', 'row', 'col']) if os.path.exists(sc) else None
        n = len(cc)
        fig = plt.figure(figsize=(11.0, 2.75 * n + 0.6))
        gs = fig.add_gridspec(n, 4, width_ratios=[1.25, 1, 1, 1], wspace=0.04, hspace=0.12,
                              left=0.01, right=0.99, top=1 - 0.6 / (2.75 * n + 0.6), bottom=0.01)
        for q, c in enumerate(cc.itertuples()):
            ndvi, b, bnd, lab, mx = panels(city, c.tile, c.row, c.col, kept)
            ax = [fig.add_subplot(gs[q, t]) for t in (1, 2, 3)]
            draw(ax, ndvi, b, bnd, mx)
            m = dict(id=c.id, city=city, tile=c.tile, row=c.row, col=c.col, n_crowns=int(len(np.unique(lab)) - 1),
                     binary_fraction=float(b.mean()))
            if sc is not None and (c.tile, c.row, c.col) in sc.index:
                s = sc.loc[(c.tile, c.row, c.col)]
                m.update(housing_density=s.housing_density, crown_ndvi=s.crown_ndvi, contrast=s.contrast, seam=s.seam)
            rows.append(m)
            t = fig.add_subplot(gs[q, 0]); t.axis('off')
            extra = f"\nhousing {m['housing_density']:.0f} units/km²" if 'housing_density' in m and np.isfinite(m['housing_density']) else ''
            desc = '\n'.join(__import__('textwrap').wrap(c.description, 30))
            t.text(0.0, 0.97, c.id, fontsize=15, fontweight='bold', va='top', transform=t.transAxes,
                   color='#B00000' if c.description.startswith('CURRENT') else 'black')
            t.text(0.0, 0.80, f'{c.tile[5:-4]}\nrow {c.row}, col {c.col}', fontsize=8, va='top', family='monospace',
                   transform=t.transAxes)
            t.text(0.0, 0.60, desc, fontsize=8.5, va='top', transform=t.transAxes)
            t.text(0.0, 0.02, f"{m['n_crowns']} crowns · binary {m['binary_fraction']:.0%}{extra}", fontsize=8,
                   va='bottom', color='#444444', transform=t.transAxes)
            if q == 0:
                for a, ttl in zip(ax, ['NDVI', 'Binary prediction (τ = 0.7)', 'Separated crowns in dataset']):
                    a.set_title(ttl, fontsize=10, fontweight='bold')
        fig.suptitle(f'Figure 6 candidates: {city.title()} (NDVI only)', fontsize=14, fontweight='bold',
                     y=1 - 0.12 / (2.75 * n + 0.6))
        for ext in ('png', 'pdf'):
            fig.savefig(f'{OUT}/Fig6_candidates_{city}.{ext}', dpi=150)
        plt.close(fig)
        print('written', city, n)
    pd.DataFrame(rows).to_csv(f'{OUT}/Fig6_candidates_metrics.csv', index=False)
