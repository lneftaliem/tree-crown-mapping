"""
01v1_tile_inventory.py -- tile inventory of the v1 dataset: exactly one NDVI + PAN prediction per in-city window.

Every one of the 347 in-city windows (code/sherlock/v1_windows.csv) was re-predicted in Sherlock job 50693038
(regen_ndvi_pan.py, NDVI band 2 + PAN band 1, threshold 0.7) and instance-separated (-cn 8 -m 15). No historical
(PAN + PAN or older-vintage) prediction enters v1. For each window this script
  * checks that the v1 prediction and split exist, that its stats line has no error, threshold 0.7 and NDVI + PAN inputs,
    and that the split raster has the prediction's grid;
  * records the window's historical file names (aliases: identical footprints under different scene names, from the
    historical PAN + PAN selection) and which of them names the v1 output;
  * links the outputs as split_links/center_based_relabeled_<name> and pred_links/<name> for the unchanged build scripts.
Writes <out>/tile_inventory.csv and <out>/window_aliases.csv.
Usage: python 01v1_tile_inventory.py <v1_download_dir> <out_dir>
"""
import os, sys, json, glob
import pandas as pd, rasterio

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.dirname(os.path.dirname(HERE))
V1, OUT = sys.argv[1:3]
CITIES = ('austin', 'bloomington', 'cupertino', 'surrey')
win = pd.read_csv(f'{P}/revision/sherlock/v1_windows.csv')
assert win.filename.is_unique and win.window.is_unique and len(win) == 347 and set(win.city) <= set(CITIES)
hist = pd.read_csv(f'{HERE}/inputs/historical_tile_inventory.csv')
hist = hist[hist.city.isin(CITIES) & (hist.decision != 'not_used_outside_city')]

stats = {}
for city in CITIES:
    for f in glob.glob(f'{V1}/stats/LN_{city}_*.jsonl'):
        for line in open(f):
            r = json.loads(line)
            if 'error' not in r or r['filename'] not in stats:
                stats[r['filename']] = r

rows, alias = [], []
for d in ('split_links', 'pred_links'):
    os.makedirs(f'{OUT}/{d}', exist_ok=True)
for _, w in win.iterrows():
    n = w.filename
    s = stats.get(n, {})
    pred, split = f'{V1}/predictions/{n}', f'{V1}/split/center_based_relabeled_{n}'
    problems = []
    if 'error' in s or not s:
        problems.append(f"stats: {s.get('error', 'missing')}")
    if s and (abs(s['thresh'] - 0.7) > 1e-9 or s['inputs'] != os.environ.get('EXPECT_INPUTS', 'ndvi_band2+pan_band1')):
        problems.append('wrong configuration')
    for f in (pred, split):
        if not os.path.exists(f):
            problems.append(f'missing {os.path.basename(f)}')
    if not problems:
        with rasterio.open(pred) as a, rasterio.open(split) as b:
            if (a.width, a.height, a.transform, a.crs) != (b.width, b.height, b.transform, b.crs):
                problems.append('split grid differs from prediction')
    h = hist[hist.files.fillna('').str.split(';').map(lambda fs: n in fs) | (hist.filename == n)]
    names = sorted(set(';'.join(h.files.dropna()).split(';')) - {''}) or [n]
    for a in names:
        alias.append(dict(window=w.window, city=w.city, filename=a, is_v1_name=a == n, n_names=len(names)))
    rows.append(dict(window=w.window, city=w.city, filename=n, mosaic=w.mosaic, L=w.L, T=w["T"], R=w.R, B=w.B,
                     n_historical_names=len(names), decision='v1_ndvi_pan' if not problems else 'v1_problem',
                     problems='; '.join(problems), pos_px=s.get('pos_px'), valid_px=s.get('valid_px'),
                     pred_source=pred, split_source=split))
    if not problems:
        for src, dst in ((split, f'{OUT}/split_links/center_based_relabeled_{n}'), (pred, f'{OUT}/pred_links/{n}')):
            if not os.path.lexists(dst):
                os.symlink(src, dst)
inv = pd.DataFrame(rows)
inv.to_csv(f'{OUT}/tile_inventory.csv', index=False)
pd.DataFrame(alias).to_csv(f'{OUT}/window_aliases.csv', index=False)
print(inv.groupby(['city', 'decision']).size().to_string())
print('windows with >1 historical name (identical footprints):', int((inv.n_historical_names > 1).sum()))
bad = inv[inv.decision != 'v1_ndvi_pan']
assert bad.empty, bad[['filename', 'problems']].to_string()
