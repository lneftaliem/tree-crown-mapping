"""
01_tile_scan.py -- full-population, tile-by-tile comparison of the original U-Net
prediction masks (prediction_tifs/used_predictions.zip) and the center-based
instance-separated rasters (split/center_based_relabeled_*.tif), plus per-crown tables.

Read-only on all inputs. Writes (under split_audit_20261031/diagnostics/):
  tile_scan.csv                 one row per tile (grid/georef checks, pixel + component counts)
  crowns/<tile>.orig.parquet    one row per ORIGINAL crown = 4-connected component of the raw
                                binary mask (the convention of prepare_ripley_data.py)
  crowns/<tile>.split.parquet   one row per SPLIT crown = one instance label value

Per-crown columns: tile, label, npx (true raster pixel count), row/col pixel centroid,
lon/lat of the centroid (EPSG:4326), edge_m (centroid distance to the nearest tile edge, m),
plus linkage columns between the two representations.

Usage:  python 01_tile_scan.py [n_workers]
"""
import os
import sys
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage as ndi
from skimage.measure import label as sk_label
from multiprocessing import Pool

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402
ROOT = paths.ROOT
SPLIT_DIR = paths.SPLIT_DIR
OUT = paths.DIAG
CROWN_DIR = f'{OUT}/crowns'
S4 = ndi.generate_binary_structure(2, 1)
S8 = ndi.generate_binary_structure(2, 2)


def city_of(name):
    for c in ('austin', 'bloomington', 'cupertino', 'surrey'):
        if f'_{c}_' in name:
            return c


def has_georef(src):
    return src.crs is not None and not (src.transform.c == 0 and src.transform.f == 0)


def label_stats(lab, n):
    """npx and pixel centroid (row, col) for labels 1..n."""
    flat = lab.ravel()
    nz = np.flatnonzero(flat)
    ids = flat[nz].astype(np.int64)
    rows, cols = np.divmod(nz, lab.shape[1])
    npx = np.bincount(ids, minlength=n + 1)
    sr = np.bincount(ids, weights=rows, minlength=n + 1)
    sc = np.bincount(ids, weights=cols, minlength=n + 1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return npx, sr / npx, sc / npx


def crown_frame(name, ids, npx, r, c, transform, shape):
    H, W = shape
    lon, lat = rasterio.transform.xy(transform, r, c)  # pixel-centre convention, as original pipeline
    lon, lat = np.asarray(lon), np.asarray(lat)
    dx_m = abs(transform.a) * 111320.0 * np.cos(np.radians(lat))
    dy_m = abs(transform.e) * 110574.0
    edge_m = np.minimum.reduce([(c + 0.5) * dx_m, (W - c - 0.5) * dx_m, (r + 0.5) * dy_m, (H - r - 0.5) * dy_m])
    return pd.DataFrame({'tile': name, 'label': ids.astype(np.int64), 'npx': npx.astype(np.int64),
                         'row': r, 'col': c, 'lon': lon, 'lat': lat, 'edge_m': edge_m})


def scan(name):
    r = {'filename': name, 'city': city_of(name)}
    with rasterio.open(paths.pred_path(name)) as so:
        o = so.read(1)
        r.update(o_dtype=so.dtypes[0], o_crs=str(so.crs), o_w=so.width, o_h=so.height,
                 o_transform=tuple(so.transform)[:6], o_max=int(o.max()), o_georef=has_georef(so))
        o_tr = so.transform
    with rasterio.open(f'{SPLIT_DIR}/center_based_relabeled_{name}') as ss:
        s = ss.read(1)
        r.update(s_dtype=ss.dtypes[0], s_crs=str(ss.crs), s_w=ss.width, s_h=ss.height,
                 s_transform=tuple(ss.transform)[:6], s_georef=has_georef(ss))
        s_tr = ss.transform
    r['same_grid'] = bool(r['o_crs'] == r['s_crs'] and r['o_w'] == r['s_w'] and r['o_h'] == r['s_h']
                          and np.allclose(r['o_transform'], r['s_transform'], rtol=0, atol=1e-12))
    transform = s_tr if r['s_georef'] else o_tr
    r['transform_source'] = 'split' if r['s_georef'] else ('orig' if r['o_georef'] else 'NONE')

    ob = o > 0
    fill = ndi.binary_fill_holes(ob)
    sb = s > 0
    r['o_px'] = int(ob.sum()); r['fill_px'] = int(fill.sum()); r['s_px'] = int(sb.sum())
    r['s_px_not_in_fill'] = int((sb & ~fill).sum())
    r['fill_px_not_in_s'] = int((fill & ~sb).sum())
    r['o_px_not_in_s'] = int((ob & ~sb).sum())

    lo4, n4 = ndi.label(ob, S4)
    _, r['o_cc8'] = ndi.label(ob, S8)
    _, r['fill_cc4'] = ndi.label(fill, S4)
    lf8, nf8 = ndi.label(fill, S8)
    r['o_cc4'] = n4; r['fill_cc8'] = nf8

    nlab = int(s.max())
    pieces8 = sk_label(s, connectivity=2, background=0)
    r['s_pieces_c8'] = int(pieces8.max())
    r['s_pieces_c4'] = int(sk_label(s, connectivity=1, background=0).max())

    # ---- split crowns -------------------------------------------------------------------------
    npx_s, rs, cs = label_stats(s, nlab)
    present = np.nonzero(npx_s[1:] > 0)[0] + 1
    r['s_nlabels'] = int(present.size); r['s_maxlabel'] = nlab
    idx = np.flatnonzero(sb.ravel())
    sv = s.ravel()[idx].astype(np.int64)
    # parent filled-8-component of every label, and #labels per parent
    pf = lf8.ravel()[idx].astype(np.int64)
    pairs = np.unique(np.stack([pf, sv], 1), axis=0)
    comps_per_label = np.bincount(pairs[:, 1], minlength=nlab + 1)
    labs_per_comp = np.bincount(pairs[:, 0], minlength=nf8 + 1)
    parent = np.zeros(nlab + 1, np.int64); parent[pairs[:, 1]] = pairs[:, 0]
    r['labels_spanning_gt1_cc8'] = int((comps_per_label > 1).sum())
    r['cc8_with_0_labels'] = int((labs_per_comp[1:] == 0).sum())
    r['cc8_split'] = int((labs_per_comp[1:] > 1).sum())
    r['cc8_extra_instances'] = int((labs_per_comp[1:][labs_per_comp[1:] > 1] - 1).sum())
    # pieces (8-connected) per label
    pv = pieces8.ravel()[idx].astype(np.int64)
    pp = np.unique(np.stack([pv, sv], 1), axis=0)
    pieces_per_label = np.bincount(pp[:, 1], minlength=nlab + 1)
    # labels spanning >1 ORIGINAL 4-connected component
    o4 = lo4.ravel()[idx].astype(np.int64)
    p4 = np.unique(np.stack([o4, sv], 1), axis=0)
    p4 = p4[p4[:, 0] > 0]
    cpl4 = np.bincount(p4[:, 1], minlength=nlab + 1)
    lpc4 = np.bincount(p4[:, 0], minlength=n4 + 1)
    r['labels_spanning_gt1_o_cc4'] = int((cpl4 > 1).sum())
    r['o_cc4_with_0_labels'] = int((lpc4[1:] == 0).sum())

    ds = crown_frame(name, present, npx_s[present], rs[present], cs[present], transform, s.shape)
    ds['parent_cc8'] = parent[present]
    ds['labels_in_parent'] = labs_per_comp[parent[present]]
    ds['parent_cc8_npx'] = np.bincount(lf8.ravel())[parent[present]]
    ds['n_pieces'] = pieces_per_label[present]
    ds['n_orig_cc4'] = cpl4[present]

    # ---- original crowns (4-connected components of the raw mask) ----------------------------
    npx_o, ro, co = label_stats(lo4, n4)
    ids = np.arange(1, n4 + 1)
    do = crown_frame(name, ids, npx_o[1:], ro[1:], co[1:], transform, o.shape)
    do['n_split_labels'] = lpc4[1:]

    r['s_label_maxpx'] = int(npx_s.max()) if nlab else 0
    r['o_cc4_maxpx'] = int(npx_o[1:].max()) if n4 else 0
    stem = name[:-4]
    ds.to_parquet(f'{CROWN_DIR}/{stem}.split.parquet', index=False)
    do.to_parquet(f'{CROWN_DIR}/{stem}.orig.parquet', index=False)
    return r


if __name__ == '__main__':
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    os.makedirs(CROWN_DIR, exist_ok=True)
    names = sorted(f.replace('center_based_relabeled_', '') for f in os.listdir(SPLIT_DIR)
                   if f.startswith('center_based_relabeled_') and f.endswith('.tif'))
    rows = []
    with Pool(nw) as pool:
        for k, r in enumerate(pool.imap_unordered(scan, names, chunksize=1)):
            rows.append(r)
            if k % 100 == 0:
                print(k, r['filename'], flush=True)
    df = pd.DataFrame(rows).sort_values('filename')
    df.to_csv(f'{OUT}/tile_scan.csv', index=False)
    print('done', len(df), 'of', len(names))
