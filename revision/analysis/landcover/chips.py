"""
chips.py -- read-only helpers for the land-cover review of detected crowns (outputs/landcover_review/).
For a v1b tile (tile_inventory filename) and a pixel window, returns the PAN (mosaic band 1) and NDVI (band 2 / 1000 - 1)
of the source mosaic warped (nearest) onto the prediction grid -- the grid the model saw -- and the instance labels,
restricted to the crowns kept in a given dataset. Nothing is written.
"""
import numpy as np, pandas as pd, rasterio, pyogrio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from affine import Affine

import os
P = os.environ.get('URBAN_DATA', '/path/to/Urban_data') + '/tree_crowns_v1_20261101'
PGC = os.environ.get('URBAN_DATA', '/path/to/Urban_data') + '/NASA:PGC/Training_06062024'
CITIES = ('austin', 'bloomington', 'cupertino', 'surrey')
INV = pd.read_csv(f'{P}/work_v1b/tiles/tile_inventory.csv').set_index('filename')
GEO = pd.read_csv(f'{P}/work_v1b/diag/georef_table.csv').set_index('filename')


def kept_labels(dataset, city, cols=('source_tile', 'instance_label')):
    k = pyogrio.read_dataframe(f'{dataset}/{city}_ripley_points_with_income.geojson', read_geometry=False, columns=list(cols))
    return k


def window(fn, r0, c0, h, w):
    """PAN, NDVI, labels for rows r0:r0+h, cols c0:c0+w of tile fn (clipped to the tile)."""
    r = INV.loc[fn]; g = GEO.loc[fn]
    assert r.city in CITIES
    px = (g.R - g.L) / g.o_w
    r0, c0 = max(0, r0), max(0, c0)
    h, w = min(h, int(g.o_h) - r0), min(w, int(g.o_w) - c0)
    tr = Affine(px, 0, g.L + c0 * px, 0, -px, g['T'] - r0 * px)
    sid = '_'.join(r.mosaic.split('_')[-3:-1])
    with rasterio.open(f'{PGC}/{r.city}/{sid}/{r.mosaic}') as m, \
            WarpedVRT(m, crs='EPSG:4326', transform=tr, width=w, height=h, resampling=Resampling.nearest) as v:
        pan, nd = v.read(1).astype(float), v.read(2).astype(float)
    nd = np.where(nd > 0, nd / 1000 - 1, np.nan)
    pan = np.where(pan > 0, pan, np.nan)
    with rasterio.open(r.split_source) as s:
        lab = s.read(1, window=((r0, r0 + h), (c0, c0 + w)))
    return pan, nd, lab, px


def stretch(a, lo=2, hi=98):
    v = a[np.isfinite(a)]
    if not len(v):
        return np.zeros_like(a)
    l, h = np.percentile(v, [lo, hi])
    return np.clip((a - l) / (h - l + 1e-9), 0, 1)
