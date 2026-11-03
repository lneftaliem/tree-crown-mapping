"""
screen_windows.py -- Figure 6 candidate screening (presentation only; read-only on all predictions, crowns and datasets).
For each v1b tile of a city, every 256 x 256 px window on a 128 px stride is described by:
  binary_fraction   share of window pixels in the binary prediction (tau = 0.7)
  n_crowns          crowns of the final dataset (tree_crowns_v1b_ndvi02) present in the window (>= 3 m2 by construction)
  coverage          share of binary-prediction pixels belonging to those crowns (low = removed / filtered components)
  n_unresolved      crowns flagged unresolved_canopy (> 250 m2)
  median_area_m2    median dataset crown area of the window's crowns
  income_share      share of the window's crowns inside an analysis unit with income
  housing_density   housing units / km2 of the block group / DA holding most of the window's crowns (v1b_final covariates)
  rt_max            max crop-row texture score of the window's 64 m blocks (work_v1b/diag/row_texture)
  veg_min           min NDVI >= 0.2 fraction of those blocks
Windows pass when binary_fraction 0.10-0.40, coverage >= 0.85, n_unresolved <= 1, income_share >= 0.9,
housing_density >= 250 km-2, rt_max <= 60, veg_min >= 0.25. Writes <out>/screen_<city>.csv (passing windows only).
Usage: python screen_windows.py <city> <out_dir> <n_workers>
"""
import sys, os
import numpy as np, pandas as pd, rasterio, pyogrio
from multiprocessing import Pool

import os
P = os.environ.get('URBAN_DATA', '/path/to/Urban_data') + '/tree_crowns_v1_20261101'
CITIES = ('austin', 'bloomington', 'cupertino', 'surrey')
CITY, OUT, NW = sys.argv[1], sys.argv[2], int(sys.argv[3])
assert CITY in CITIES
W, S = 256, 128
INV = pd.read_csv(f'{P}/work_v1b/tiles/tile_inventory.csv')
INV = INV[(INV.city == CITY) & (INV.decision == 'v1_ndvi_pan')].set_index('filename')
GEO = pd.read_csv(f'{P}/work_v1b/diag/georef_table.csv').set_index('filename')
U = pd.read_csv(f'{P}/outputs/runs/v1b_final/analysis_output/equity_results/all_cities_urban_covariates_corrected.csv')
HD = U[U.city == CITY].set_index('clean_id').housing_density_km2
K = pyogrio.read_dataframe(f'{P}/dataset/tree_crowns_v1b_ndvi02/{CITY}_ripley_points_with_income.geojson', read_geometry=False,
                           columns=['source_tile', 'instance_label', 'crown_area_m2', 'unresolved_canopy', 'block_group_id',
                                    'has_income'])
KT = {t: g.set_index(g.instance_label.astype('int64')) for t, g in K.groupby('source_tile')}


def run(fn):
    k = KT.get(fn)
    if k is None:
        return None
    r = INV.loc[fn]
    with rasterio.open(r.split_source) as s:
        lab = s.read(1)
    with rasterio.open(r.pred_source) as p:
        pr = p.read(1) > 0
    n_lab = int(lab.max()) + 1
    inds = np.zeros(n_lab, bool); ids = k.index.values[k.index.values < n_lab]; inds[ids] = True
    area = np.zeros(n_lab); area[ids] = k.loc[ids, 'crown_area_m2'].values
    unres = np.zeros(n_lab, bool); unres[ids] = k.loc[ids, 'unresolved_canopy'].astype(str).str.lower().isin(['true', '1', 'yes']).values
    inc = np.zeros(n_lab, bool); inc[ids] = (k.loc[ids, 'has_income'] == 'yes').values
    bg = np.zeros(n_lab, 'int64'); bg[ids] = k.loc[ids, 'block_group_id'].fillna(0).astype('int64').values
    rt = None
    f = f'{P}/work_v1b/diag/row_texture/blocks_{fn}.parquet'
    if os.path.exists(f):
        rt = pd.read_parquet(f).drop_duplicates(['block_r', 'block_c']).set_index(['block_r', 'block_c'])
    H, Wd = lab.shape
    ip = np.pad(pr.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    rows = []
    for i in range(0, H - W + 1, S):
        for j in range(0, Wd - W + 1, S):
            bf = (ip[i + W, j + W] - ip[i, j + W] - ip[i + W, j] + ip[i, j]) / W ** 2
            if not 0.10 <= bf <= 0.40:
                continue
            l = lab[i:i + W, j:j + W]
            kept_px = inds[l] & (l > 0)
            cov = kept_px.sum() / max(pr[i:i + W, j:j + W].sum(), 1)
            if cov < 0.85:
                continue
            u = np.unique(l[kept_px])
            if len(u) < 20:
                continue
            if rt is not None:
                b = rt.reindex([(i // 128 + a, j // 128 + c) for a in (0, 1) for c in (0, 1)])
                rtm, vmin = float(b.score.max()), float(b.veg_frac.min())
            else:
                rtm, vmin = np.nan, np.nan
            bgs, cnt = np.unique(bg[u], return_counts=True)
            mbg = int(bgs[cnt.argmax()])
            rows.append(dict(city=CITY, tile=fn, row=i, col=j, binary_fraction=bf, n_crowns=len(u), coverage=cov,
                             n_unresolved=int(unres[u].sum()), median_area_m2=float(np.median(area[u])),
                             income_share=float(inc[u].mean()), block_group=mbg, housing_density=float(HD.get(mbg, np.nan)),
                             rt_max=rtm, veg_min=vmin))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    with Pool(NW) as pool:
        parts = [d for d in pool.imap_unordered(run, sorted(INV.index)) if d is not None and len(d)]
    D = pd.concat(parts, ignore_index=True)
    ok = ((D.n_unresolved <= 1) & (D.income_share >= 0.9) & (D.housing_density >= 250) & (D.rt_max <= 60)
          & (D.veg_min >= 0.25))
    print(CITY, 'windows evaluated (after binary/coverage/>=20 crowns):', len(D), 'passing:', int(ok.sum()))
    D[ok].sort_values(['n_crowns', 'tile', 'row', 'col'], ascending=[False, True, True, True]).to_csv(
        f'{OUT}/screen_{CITY}.csv', index=False)
