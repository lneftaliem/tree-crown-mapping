"""
17_crown_ndvi.py -- DIAGNOSTIC: mean NDVI under every instance-separated crown of the v1 tiles (does not change the dataset).

For each v1 tile, the source mosaic (band 2 = (NDVI + 1) * 1000, band 1 = PAN; local copy NASA:PGC/Training_06062024/<city>/
<scene>/) is warped (nearest) onto the tile's prediction grid (EPSG:4326, embedded transform) and, per instance label of the
split raster, the mean NDVI, the share of pixels with NDVI < 0.1 and the pixel count are computed.
Writes DIAG_DIR/crown_ndvi/<tile>.parquet (tile, label, npx, ndvi_mean, share_ndvi_lt01).
Usage: DIAG_DIR=... python 17_crown_ndvi.py <tile_inventory.csv> [n_workers]
"""
import os, sys
import numpy as np, pandas as pd, rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from multiprocessing import Pool
from types import SimpleNamespace

PGC = '/path/to/Urban_data/NASA:PGC/Training_06062024'
OUT = os.path.join(os.environ['DIAG_DIR'], 'crown_ndvi')
CITIES = ('austin', 'bloomington', 'cupertino', 'surrey')


def run(r):
    dst = f'{OUT}/{r.filename[:-4]}.parquet'
    if os.path.exists(dst):
        return
    sid = '_'.join(r.mosaic.split('_')[-3:-1])
    with rasterio.open(r.split_source) as s:
        lab = s.read(1); tr, w, h, crs = s.transform, s.width, s.height, s.crs
    with rasterio.open(f'{PGC}/{r.city}/{sid}/{r.mosaic}') as m, \
            WarpedVRT(m, crs=crs, transform=tr, width=w, height=h, resampling=Resampling.nearest) as v:
        raw = v.read(2).astype(np.float64)
    ndvi = raw / 1000 - 1
    ok = (lab > 0) & (raw > 0)
    L = lab[ok].astype(np.int64)
    n = np.bincount(L)
    s = np.bincount(L, weights=ndvi[ok])
    lo = np.bincount(L, weights=(ndvi[ok] < 0.1).astype(float))
    idx = np.nonzero(n)[0]
    pd.DataFrame(dict(tile=r.filename, label=idx, npx=n[idx], ndvi_mean=s[idx] / n[idx],
                      share_ndvi_lt01=lo[idx] / n[idx])).to_parquet(dst, index=False)


if __name__ == '__main__':
    inv = pd.read_csv(sys.argv[1])
    assert set(inv.city) <= set(CITIES)
    os.makedirs(OUT, exist_ok=True)
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 6) as p:
        p.map(run, [SimpleNamespace(**r) for r in inv.to_dict("records")], chunksize=1)
    print('done', len(os.listdir(OUT)))
