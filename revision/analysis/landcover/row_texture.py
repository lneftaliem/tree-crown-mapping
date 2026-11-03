"""
row_texture.py -- wall-to-wall row-crop texture indicator for every v1b tile (read-only on all data).
For each non-overlapping 128 x 128 px block (64 m) of a tile, on the PAN band (mosaic band 1) warped onto the
prediction grid: periodicity score = peak power / mean power of the 2-D spectrum (Hann window) in the annulus of
periods 3-12 px (1.5-6 m), x 0.5 for the symmetric pair. Regular crop rows give scores of about 200-480 and
residential/woodland chips 10-70 (tested on the Fig 6 Surrey field and 31 random Surrey crowns). Also block vegetated
fraction (NDVI >= 0.2) and mean NDVI. Each label of the tile is attached to the block of its pixel centroid.
Writes <out>/<tile>.parquet: tile, label, npx, block_r, block_c, score, veg_frac, ndvi_block.
Usage: python row_texture.py <out_dir> <n_workers>
"""
import sys, os
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chips

B = 128
fr = np.hypot(np.fft.fftfreq(B)[:, None], np.fft.fftfreq(B)[None, :])
ANN = (fr >= 1 / 12) & (fr <= 1 / 3)
HANN = np.outer(np.hanning(B), np.hanning(B))
OUT = sys.argv[1]


def run(fn):
    o = f'{OUT}/{fn}.parquet'
    if os.path.exists(o):
        return fn
    g = chips.GEO.loc[fn]
    pan, nd, lab, _ = chips.window(fn, 0, 0, int(g.o_h), int(g.o_w))
    H, W = lab.shape
    nb_r, nb_c = H // B, W // B
    sc = np.full((nb_r, nb_c), np.nan); vf = np.full_like(sc, np.nan); nm = np.full_like(sc, np.nan)
    for i in range(nb_r):
        for j in range(nb_c):
            p = pan[i * B:(i + 1) * B, j * B:(j + 1) * B]; n = nd[i * B:(i + 1) * B, j * B:(j + 1) * B]
            ok = np.isfinite(p)
            if ok.mean() < 0.9:
                continue
            a = np.where(ok, p, np.nanmean(p)); a = (a - a.mean()) / (a.std() + 1e-9)
            P = np.abs(np.fft.fft2(a * HANN)) ** 2
            v = P[ANN]; sc[i, j] = v.max() / v.sum() * len(v) / 2
            vf[i, j] = np.nanmean(n >= 0.2); nm[i, j] = np.nanmean(n)
    L = lab.ravel(); m = L > 0
    rr, cc = np.divmod(np.flatnonzero(m), W)
    lm = L[m]; n_lab = int(lm.max()) + 1 if len(lm) else 1
    cnt = np.bincount(lm, minlength=n_lab)
    r_mean = np.bincount(lm, rr, n_lab) / np.maximum(cnt, 1); c_mean = np.bincount(lm, cc, n_lab) / np.maximum(cnt, 1)
    ids = np.flatnonzero(cnt)
    br = np.clip((r_mean[ids] // B).astype(int), 0, nb_r - 1); bc = np.clip((c_mean[ids] // B).astype(int), 0, nb_c - 1)
    pd.DataFrame(dict(tile=fn, label=ids, npx=cnt[ids], block_r=br, block_c=bc, score=sc[br, bc], veg_frac=vf[br, bc],
                      ndvi_block=nm[br, bc])).to_parquet(o)
    pd.DataFrame(dict(tile=fn, block_r=np.repeat(np.arange(nb_r), nb_c), block_c=np.tile(np.arange(nb_c), nb_r),
                      score=sc.ravel(), veg_frac=vf.ravel(), ndvi_block=nm.ravel())).to_parquet(f'{OUT}/blocks_{fn}.parquet')
    return fn


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    tiles = [f for f in chips.INV.index if chips.INV.loc[f, 'city'] in chips.CITIES]
    with Pool(int(sys.argv[2])) as p:
        for k, fn in enumerate(p.imap_unordered(run, tiles), 1):
            if k % 20 == 0:
                print(k, '/', len(tiles), flush=True)
    print('done', len(tiles))
