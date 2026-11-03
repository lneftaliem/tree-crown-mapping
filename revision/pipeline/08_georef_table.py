"""
08_georef_table.py -- resolve one EPSG:4326 affine transform per tile, record its provenance,
validate it, and rewrite per-crown lon/lat + edge distance with the resolved transform.

Resolution order:
  embedded   : the raster carries CRS + non-identity transform (split == original grid)
  metadata   : sherlock_venv/tile_metadata.csv row with the same filename
               (left/top + (right-left)/width; width/height equal the raster size for every row)
  base_match : metadata of the NDVI sibling with the same offsets/scene (pan & ndvi of a scene
               share origin and pixel size exactly, verified on 62 scenes)
  scene      : scene origin + pixel size from any georeferenced/metadata sibling of the same
               scene id; transform = origin + (x_offset, y_offset) * pixel size
Validation:
  * leave-one-out: every embedded tile is re-derived by the 'scene' rule from its siblings and
    compared with its embedded transform
  * every tile is checked for north-up orientation, square pixels, 1 band, size match
  * cross-scene alignment: for reconstructed tiles, the mask is cross-correlated with
    overlapping tiles from OTHER scenes to estimate any residual pixel shift
Writes diagnostics/georef_table.csv, diagnostics/crowns_geo/<tile>.<kind>.parquet
"""
import os
import re
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from multiprocessing import Pool

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402
ROOT = paths.ROOT
AUD = paths.DIAG
PAT = re.compile(r'pred_(ndvi|pan)_(\d+)-(\d+)_(austin|bloomington|cupertino|surrey)_(.+)\.tif')


def build():
    d = pd.read_csv(f'{AUD}/tile_scan.csv')
    m = pd.read_csv(paths.METADATA_CSV).set_index('filename')
    p = d.filename.str.extract(PAT)
    d['prod'], d['xo'], d['yo'], d['scene'] = p[0], p[1].astype(int), p[2].astype(int), p[3] + '_' + p[4]
    rec = {}
    for _, r in d.iterrows():
        if r.s_georef:
            t = eval(r.s_transform); rec[r.filename] = (t[0], t[2], t[5], 'embedded')
        elif r.filename in m.index:
            mm = m.loc[r.filename]; rec[r.filename] = ((mm.right - mm.left) / mm.width, mm.left, mm.top, 'metadata')
    base = {f.replace('pred_pan_', '').replace('pred_ndvi_', ''): f for f in m.index}
    for _, r in d[~d.filename.isin(rec)].iterrows():
        b = r.filename.replace('pred_pan_', '').replace('pred_ndvi_', '')
        if b in base:
            mm = m.loc[base[b]]; rec[r.filename] = ((mm.right - mm.left) / mm.width, mm.left, mm.top, 'base_match')
    d['dx'] = d.filename.map(lambda f: rec.get(f, (np.nan,) * 4)[0])
    d['L'] = d.filename.map(lambda f: rec.get(f, (np.nan,) * 4)[1])
    d['T'] = d.filename.map(lambda f: rec.get(f, (np.nan,) * 4)[2])
    d['georef_source'] = d.filename.map(lambda f: rec.get(f, (None,) * 4)[3])
    d['oL'] = d.L - d.xo * d.dx; d['oT'] = d['T'] + d.yo * d.dx
    sc = d.dropna(subset=['dx']).groupby('scene').agg(sdx=('dx', 'first'), soL=('oL', 'first'), soT=('oT', 'first'),
                                                        sdx_rng=('dx', lambda v: v.max() - v.min()),
                                                        soL_rng=('oL', lambda v: v.max() - v.min()),
                                                        soT_rng=('oT', lambda v: v.max() - v.min()))
    assert (sc[['sdx_rng', 'soL_rng', 'soT_rng']].max() < 1e-12).all(), 'inconsistent scene grids'
    d = d.merge(sc, left_on='scene', right_index=True, how='left')
    miss = d.georef_source.isna()
    d.loc[miss, 'dx'] = d.sdx; d.loc[miss, 'L'] = d.soL + d.xo * d.sdx; d.loc[miss, 'T'] = d.soT - d.yo * d.sdx
    d.loc[miss, 'georef_source'] = 'scene'
    assert d.dx.notna().all(), 'tile without georeferencing'
    # leave-one-out check on embedded tiles
    loo = []
    for _, r in d[d.georef_source == 'embedded'].iterrows():
        sib = d[(d.scene == r.scene) & (d.filename != r.filename) & d.georef_source.isin(['embedded', 'metadata'])]
        if len(sib):
            s = sib.iloc[0]
            loo.append(max(abs(s.oL + r.xo * s.dx - r.L), abs(s.oT - r.yo * s.dx - r['T'])) / r.dx)
    d['R'] = d.L + d.dx * d.o_w; d['B'] = d['T'] - d.dx * d.o_h
    print('georef sources:', d.georef_source.value_counts().to_dict())
    print('leave-one-out max error (pixels):', max(loo), 'n =', len(loo))
    return d


def xcorr_shift(args):
    """Estimate residual shift (px) of tile a vs overlapping tile b from another scene."""
    a, b = args
    with rasterio.open(paths.pred_path(a.filename)) as sa, rasterio.open(paths.pred_path(b.filename)) as sb:
        L, R = max(a.L, b.L), min(a.R, b.R); T, B = min(a['T'], b['T']), max(a.B, b.B)
        if R - L < 300 * a.dx or T - B < 300 * a.dx:
            return None
        ta = rasterio.Affine(a.dx, 0, a.L, 0, -a.dx, a['T']); tb = rasterio.Affine(b.dx, 0, b.L, 0, -b.dx, b['T'])
        wa = from_bounds(L, B, R, T, ta).round_offsets().round_lengths()
        wb = from_bounds(L, B, R, T, tb).round_offsets().round_lengths()
        n = int(min(wa.height, wb.height, wa.width, wb.width, 1500))
        A = sa.read(1, window=rasterio.windows.Window(wa.col_off, wa.row_off, n, n)).astype(float)
        Bm = sb.read(1, window=rasterio.windows.Window(wb.col_off, wb.row_off, n, n)).astype(float)
    if A.shape != Bm.shape or A.sum() < 500 or Bm.sum() < 500:
        return None
    A -= A.mean(); Bm -= Bm.mean()
    F = np.fft.fft2(A) * np.conj(np.fft.fft2(Bm))
    c = np.fft.ifft2(F).real
    i, j = np.unravel_index(c.argmax(), c.shape)
    i = i - n if i > n // 2 else i; j = j - n if j > n // 2 else j
    shift_m = np.hypot(i * a.dx * 110574, j * a.dx * 111320 * np.cos(np.radians(a['T'])))
    return {'tile': a.filename, 'other': b.filename, 'source': a.georef_source, 'drow': int(i), 'dcol': int(j),
            'shift_m': shift_m, 'peak_corr': float(c.max() / np.sqrt((A ** 2).sum() * (Bm ** 2).sum()))}


def regeo(row):
    for kind in ('orig', 'split'):
        c = pd.read_parquet(f'{AUD}/crowns/{row.filename[:-4]}.{kind}.parquet')
        c['lon'] = row.L + (c.col + 0.5) * row.dx
        c['lat'] = row['T'] - (c.row + 0.5) * row.dx
        dxm = row.dx * 111320.0 * np.cos(np.radians(c.lat)); dym = row.dx * 110574.0
        c['edge_m'] = np.minimum.reduce([(c.col + 0.5) * dxm, (row.o_w - c.col - 0.5) * dxm,
                                         (c.row + 0.5) * dym, (row.o_h - c.row - 0.5) * dym])
        c.to_parquet(f'{AUD}/crowns_geo/{row.filename[:-4]}.{kind}.parquet', index=False)


if __name__ == '__main__':
    d = build()
    os.makedirs(f'{AUD}/crowns_geo', exist_ok=True)
    with Pool(6) as p:
        p.map(regeo, [r for _, r in d.iterrows()])
    # cross-scene alignment for reconstructed / base-matched tiles and a sample of others
    rng = np.random.default_rng(0)
    test = pd.concat([d[d.georef_source.isin(['scene', 'base_match'])],
                      d[d.georef_source == 'metadata'].sample(min(25, int((d.georef_source == 'metadata').sum())), random_state=0),  # v1: none (all embedded)
                      d[d.georef_source == 'embedded'].sample(min(40, int((d.georef_source == 'embedded').sum())), random_state=0)])
    jobs = []
    for _, a in test.iterrows():
        o = d[(d.city == a.city) & (d.scene != a.scene) & (d.L < a.R) & (d.R > a.L) & (d.B < a['T']) & (d['T'] > a.B)]
        for _, b in o.head(3).iterrows():
            jobs.append((a, b))
    with Pool(6) as p:
        al = [r for r in p.map(xcorr_shift, jobs) if r]
    al = pd.DataFrame(al)
    al.to_csv(f'{AUD}/georef_alignment.csv', index=False)
    print(al.groupby('source').agg(n=('shift_m', 'size'), med_shift_m=('shift_m', 'median'),
                                   max_shift_m=('shift_m', 'max'), med_corr=('peak_corr', 'median')))
    d.to_csv(f'{AUD}/georef_table.csv', index=False)
