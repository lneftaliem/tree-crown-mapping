"""
completeness.py -- detection completeness per census unit (read-only on all data).
For every v1b tile, on a 4x-decimated grid (2 m pixels; nearest sampling of the prediction grid): the source mosaic NDVI
(band 2), the kept-crown mask (instance labels that are in the given dataset), the census unit (exact block groups /
DAs, rasterized) and the municipality. Per (tile, unit), inside the municipality: pixel counts of NDVI >= 0.4 / 0.6,
of kept crowns, of kept crowns with NDVI >= 0.6, and of 'closed-canopy' pixels (NDVI >= 0.6 in a 64 m block whose
NDVI >= 0.6 share is >= 70 %) and of those covered by kept crowns. Overlapping tile windows are counted in each tile
(rates, not areas, are used). Writes <out>/<tile>.parquet.
Usage: python completeness.py <dataset> <boundaries_dir> <out_dir> <n_workers>
"""
import sys, os
import numpy as np, pandas as pd, geopandas as gpd, rasterio, pyogrio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.features import rasterize
from affine import Affine
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chips

DS, BND, OUT, NW = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
D = 4  # decimation
_cache = {}


def city_data(city):
    if city not in _cache:
        k = pyogrio.read_dataframe(f'{DS}/{city}_ripley_points_with_income.geojson', read_geometry=False,
                                   columns=['source_tile', 'instance_label'])
        kept = {t: np.asarray(sorted(g.instance_label.astype('int64'))) for t, g in k.groupby('source_tile')}
        bg = gpd.read_file(f'{BND}/{city}_exact_block_groups.geojson').to_crs(4326)
        bg['uid'] = bg['DAUID' if city == 'surrey' else 'GEOID'].astype('int64')
        muni = gpd.read_file(f'{BND}/{city}_exact_boundary.geojson').to_crs(4326)
        _cache[city] = (kept, bg, muni)
    return _cache[city]


def run(fn):
    o = f'{OUT}/{fn}.parquet'
    if os.path.exists(o):
        return fn
    r = chips.INV.loc[fn]; g = chips.GEO.loc[fn]
    kept, bg, muni = city_data(r.city)
    H, W = int(g.o_h) // D, int(g.o_w) // D
    px = (g.R - g.L) / g.o_w
    tr = Affine(px * D, 0, g.L, 0, -px * D, g['T'])
    sid = '_'.join(r.mosaic.split('_')[-3:-1])
    with rasterio.open(f'{chips.PGC}/{r.city}/{sid}/{r.mosaic}') as m, \
            WarpedVRT(m, crs='EPSG:4326', transform=tr, width=W, height=H, resampling=Resampling.nearest) as v:
        nd = v.read(2).astype(float)
    valid = nd > 0; nd = nd / 1000 - 1
    with rasterio.open(r.split_source) as s:
        lab = s.read(1, out_shape=(H, W), resampling=Resampling.nearest)
    ks = kept.get(fn, np.array([], 'int64'))
    crown = np.isin(lab, ks) & (lab > 0)
    unit = rasterize(((geom, i + 1) for i, geom in enumerate(bg.geometry)), out_shape=(H, W), transform=tr, fill=0, dtype='int32')
    inside = rasterize(((geom, 1) for geom in muni.geometry), out_shape=(H, W), transform=tr, fill=0, dtype='uint8').astype(bool)
    g6 = (nd >= 0.6) & valid
    # 64 m blocks = 32 decimated px
    B = 32; hb, wb = H // B * B, W // B * B
    blk = np.zeros_like(g6)
    share = g6[:hb, :wb].reshape(hb // B, B, wb // B, B).mean(axis=(1, 3))
    blk[:hb, :wb] = np.repeat(np.repeat(share >= 0.7, B, 0), B, 1)
    closed = g6 & blk
    m = inside & valid & (unit > 0)
    u = unit[m] - 1
    def cnt(x):
        return np.bincount(u, weights=x[m].astype(float), minlength=len(bg))
    n = np.bincount(u, minlength=len(bg))
    df = pd.DataFrame(dict(tile=fn, city=r.city, uid=bg.uid.values, n_px=n, n_ndvi04=cnt((nd >= 0.4) & valid), n_ndvi06=cnt(g6),
                           n_crown=cnt(crown), n_crown_ndvi06=cnt(crown & g6), n_closed=cnt(closed), n_crown_closed=cnt(crown & closed)))
    df[df.n_px > 0].to_parquet(o)
    return fn


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    tiles = sorted(f for f in chips.INV.index if chips.INV.loc[f, 'city'] in chips.CITIES)
    tiles.sort(key=lambda f: chips.INV.loc[f, 'city'])
    with Pool(NW) as p:
        for k, fn in enumerate(p.imap(run, tiles), 1):
            if k % 25 == 0:
                print(k, '/', len(tiles), flush=True)
    print('done', len(tiles))
