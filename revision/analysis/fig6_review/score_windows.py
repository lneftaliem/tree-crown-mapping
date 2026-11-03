"""
score_windows.py -- Figure 6 candidate scoring, second stage (presentation only; read-only). For the windows passing
screen_windows.py, adds NDVI-display and artifact measures, computed on the NDVI band (mosaic band 2) warped onto the
prediction grid (chips.window) and on the binary prediction:
  crown_ndvi    mean NDVI of the window's dataset-crown pixels
  bg_ndvi       mean NDVI of window pixels outside the binary prediction
  contrast      crown_ndvi - bg_ndvi (crowns stand out in the NDVI panel)
  bare          share of pixels with NDVI < 0.1 (pavement, roofs, bare ground, water)
  green_unpred  share of pixels with NDVI >= 0.5 outside the binary prediction (lawn, turf, fields, undetected canopy)
  seam          longest straight vertical / horizontal prediction edge, px: the longest run of consecutive rows (columns)
                in which the binary prediction changes between the same two adjacent columns (rows). Straight edges of
                >= 30 px (15 m) are patch-boundary or road-edge artefacts that read poorly in a figure.
Writes <out>/scored_<city>.csv.
Usage: python score_windows.py <city> <out_dir> <n_workers>
"""
import sys, os
import numpy as np, pandas as pd, rasterio
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'landcover'))
import chips

CITY, OUT, NW = sys.argv[1], sys.argv[2], int(sys.argv[3])
assert CITY in chips.CITIES
W = 256
D = pd.read_csv(f'{OUT}/screen_{CITY}.csv')
K = chips.kept_labels(f'{chips.P}/dataset/tree_crowns_v1b_ndvi02', CITY)
KEPT = {t: np.asarray(sorted(set(g.instance_label.astype('int64')))) for t, g in K.groupby('source_tile')}


def longest_run(e):
    """longest run of True down axis 0, over all columns"""
    best, cur = 0, np.zeros(e.shape[1], int)
    for row in e:
        cur = np.where(row, cur + 1, 0); best = max(best, int(cur.max()))
    return best


def run(fn):
    d = D[D.tile == fn]
    g = chips.GEO.loc[fn]
    _, nd, lab, _ = chips.window(fn, 0, 0, int(g.o_h), int(g.o_w))
    with rasterio.open(chips.INV.loc[fn].pred_source) as p:
        pr = p.read(1) > 0
    inds = np.zeros(int(lab.max()) + 1, bool); k = KEPT[fn]; inds[k[k < len(inds)]] = True
    out = []
    for r in d.itertuples():
        sl = np.s_[r.row:r.row + W, r.col:r.col + W]
        n, b, l = nd[sl], pr[sl], lab[sl]
        c = inds[l] & (l > 0)
        ok = np.isfinite(n)
        sv = max(longest_run(b[:, 1:] != b[:, :-1]), longest_run((b[1:] != b[:-1]).T))
        out.append(dict(tile=fn, row=r.row, col=r.col, crown_ndvi=float(np.nanmean(n[c])), bg_ndvi=float(np.nanmean(n[~b])),
                        bare=float(np.mean(n[ok] < 0.1)), green_unpred=float(np.mean((n >= 0.5) & ~b & ok) / max(ok.mean(), 1e-9)),
                        valid=float(ok.mean()), seam=sv))
    return pd.DataFrame(out)


if __name__ == '__main__':
    with Pool(NW) as pool:
        S = pd.concat(pool.imap_unordered(run, sorted(D.tile.unique())), ignore_index=True)
    S = D.merge(S, on=['tile', 'row', 'col'])
    S['contrast'] = S.crown_ndvi - S.bg_ndvi
    S.to_csv(f'{OUT}/scored_{CITY}.csv', index=False)
    print(CITY, len(S), S[['crown_ndvi', 'bg_ndvi', 'contrast', 'bare', 'green_unpred', 'seam']].describe().round(2).T.to_string())
