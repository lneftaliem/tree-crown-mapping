"""
18_norm_test_manifest.py -- manifests for the inference-normalization test (option B of reports/ndvi_false_positive_diagnostic.md).

  patches: the validation patches that can be re-extracted from a local mosaic (same matching rule as
           reports/validation_patch_ndvi.csv: patch centre inside the mosaic footprint, >90 % valid pixels on the patch grid).
           Columns: frame (index in threshold_v1 frame_order.txt), patch, city, mosaic, left, bottom, right, top.
  tiles:   the four Figure 6 tiles (non-vegetated surfaces) + 2 random v1 tiles per city (seed 0); v1 chunk-manifest lines.
Usage: python 18_norm_test_manifest.py <out_dir>
"""
import os, sys, glob
import numpy as np, pandas as pd, rasterio
from rasterio.vrt import WarpedVRT
from affine import Affine
from pyproj import Transformer

U = os.environ.get('URBAN_DATA', '/path/to/Urban_data')
PGC = f'{U}/NASA:PGC/Training_06062024'
P = f'{U}/tree_crowns_v1_20261101'
CITIES = ('austin', 'bloomington', 'cupertino', 'surrey')
out = sys.argv[1]; os.makedirs(out, exist_ok=True)

meta = pd.read_csv(f'{U}/sherlock_venv/tile_metadata.csv'); meta = meta[meta.filename.str.contains('_patch_')]
frames = [l.strip() for l in open(f'{U}/threshold_v1_20261101/bin_0.7/frame_order.txt')]
mos = {}
for c in CITIES:
    for f in sorted(glob.glob(f'{PGC}/{c}/*/LN_{c}_*_PV_*_mosaic.tif')):
        with rasterio.open(f) as s:
            mos[f] = (c, s.crs, s.bounds)
rows = []
for i, b in enumerate(frames):
    city = b.split('_')[1]
    if city not in CITIES:
        continue
    m = meta[meta.filename == f'pred_ndvi_0-0_{b.replace("ndvi_", "").replace(".png", "")}.tif']
    if m.empty:
        continue
    m = m.iloc[0]
    tr = Affine((m.right - m.left) / 256, 0, m.left, 0, -(m.top - m.bottom) / 256, m.top)
    for f, (c, crs, bd) in mos.items():
        if c != city:
            continue
        x, y = Transformer.from_crs(4326, crs, always_xy=True).transform(m.center_lon, m.center_lat)
        if not (bd.left < x < bd.right and bd.bottom < y < bd.top):
            continue
        with rasterio.open(f) as s, WarpedVRT(s, crs='EPSG:4326', transform=tr, width=256, height=256) as v:
            if (v.read(2) > 0).mean() > 0.9:
                rows.append(dict(frame=i, patch=b, city=city, mosaic=os.path.basename(f),
                                 left=m.left, bottom=m.bottom, right=m.right, top=m.top))
                break
pd.DataFrame(rows).to_csv(f'{out}/patches.csv', index=False)
print('patches', len(rows), pd.DataFrame(rows).city.value_counts().to_dict())

fig6 = pd.read_csv(f'{P}/outputs/runs/v1_final_fig6/Fig6_windows.csv').tile.tolist()
inv = pd.read_csv(f'{P}/work_v1/tiles/tile_inventory.csv')
inv = inv[(inv.decision == 'v1_ndvi_pan') & ~inv.filename.isin(fig6)]
rnd = [g.sample(2, random_state=0).filename.tolist() for _, g in inv.groupby('city')]
pick = {f: 'fig6' for f in fig6} | {f: 'random' for fs in rnd for f in fs}
lines = {}
for fn in sorted(glob.glob(f'{P}/code/sherlock/v1_chunks/v1_*.txt')):
    for l in open(fn):
        if l.split() and l.split()[0] in pick:
            lines[l.split()[0]] = l.strip()
assert set(lines) == set(pick), set(pick) - set(lines)
with open(f'{out}/tiles.txt', 'w') as o:
    for f in pick:
        o.write(lines[f] + '\n')
pd.Series(pick, name='role').rename_axis('filename').to_csv(f'{out}/tile_roles.csv')
print('tiles', pick)
