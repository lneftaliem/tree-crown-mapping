"""
norm_test.py -- TEST ONLY (does not touch v1 outputs): effect of the inference normalization extent on NDVI+PAN predictions.

Same model, inputs (NDVI = band 2, PAN = band 1 of the raw PV mosaic), 256-px windows, stride 224 and MAX merge as
regen_ndvi_pan.py (unet_infer.detect_tree, verbatim). Variants differ only in how each channel is standardized:
  P  per patch, image_normalize(axis=(0,1)) inside detect_tree (the original / v1 inference)
  T  per 5000-px tile: (x - mean) / std over the tile's valid pixels (PAN > 0 and NDVI raw > 0); nodata set to 0
  S  per scene: the same with mean / std over the whole mosaic's valid pixels (1/8-decimated read)
Mode 'patches': every re-extractable validation patch (18_norm_test_manifest.py patches.csv), on the patch's own EPSG:4326
  grid (WarpedVRT, nearest); T statistics from a 5000 x 5000 context centred on the patch; inference on a 768 x 768 crop,
  centre 256 x 256 kept; plus P256 = the patch alone, as in the validation protocol. Writes <out>/patches_prob.npz.
Mode 'tiles': whole v1 tiles (tiles.txt, v1 chunk-manifest lines) on their v1 grid (WarpedVRT, nearest);
  writes <out>/tiles/<variant>_<name> (uint8 round(255 p)).
ORDER=b1b2 swaps the two input channels (channel 0 = band 1, channel 1 = band 2) and also the scene statistics accordingly;
the 'ndvi'/'pan' variable names then refer to channel 0 / channel 1.
Usage: python norm_test.py patches <patches.csv> <out>  |  python norm_test.py tiles <tiles.txt> <first> <n> <out>
"""
import os, sys, time
import numpy as np, pandas as pd, rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from affine import Affine

RAW = os.environ['RAW_DIR']
ORDER = os.environ.get('ORDER', 'b2b1')  # b2b1 = v1 (detect_tree(ndvi=band 2, pan=band 1)); b1b2 = training order
CITIES = ('austin', 'bloomington', 'cupertino', 'surrey')
STRIDE, CTX, CROP = 224, 5000, 768

import tensorflow as tf
n = int(os.environ.get('SLURM_CPUS_PER_TASK', '4'))
tf.config.threading.set_intra_op_parallelism_threads(n); tf.config.threading.set_inter_op_parallelism_threads(2)
sys.path.insert(0, os.environ['NB_DIR'])
from core.losses import tversky, accuracy, dice_coef, dice_loss, specificity, sensitivity
from core.frame_info import image_normalize
from tensorflow.keras.models import load_model
import unet_infer
unet_infer.model = load_model(os.environ['MODEL'], custom_objects={'tversky': tversky, 'dice_coef': dice_coef,
                              'dice_loss': dice_loss, 'accuracy': accuracy, 'specificity': specificity,
                              'sensitivity': sensitivity}, compile=False)

_scene = {}


def scene_stats(mosaic):
    if mosaic not in _scene:
        with rasterio.open(f'{RAW}/{mosaic}') as s:
            a = s.read([1, 2], out_shape=(2, s.height // 8, s.width // 8), resampling=Resampling.nearest).astype(np.float64)
        ok = (a[0] > 0) & (a[1] > 0)
        _scene[mosaic] = [(a[k][ok].mean(), a[k][ok].std()) for k in ((0, 1) if ORDER == 'b1b2' else (1, 0))]  # (channel 0, channel 1)
    return _scene[mosaic]


def read(mosaic, tr, w, h):
    with rasterio.open(f'{RAW}/{mosaic}') as s, \
            WarpedVRT(s, crs='EPSG:4326', transform=tr, width=w, height=h, resampling=Resampling.nearest) as v:
        pan, ndvi = v.read(1).astype(np.float32), v.read(2).astype(np.float32)
    if ORDER == 'b1b2':  # model channel 0 = band 1, channel 1 = band 2 (the order of the training / validation PNGs)
        return pan, ndvi
    return ndvi, pan


def standardize(ndvi, pan, stats):
    ok = (pan > 0) & (ndvi > 0)
    out = []
    for x, (m, s) in zip((ndvi, pan), stats):
        y = ((x - m) / (s + 1e-8)).astype(np.float32); y[~ok] = 0; out.append(y)
    return out


def valid_stats(ndvi, pan):
    ok = (pan > 0) & (ndvi > 0)
    return [(float(x[ok].mean()), float(x[ok].std())) for x in (ndvi, pan)]


def run(ndvi, pan, variant, stats=None):
    if variant == 'P':
        return unet_infer.detect_tree(ndvi, pan, width=256, height=256, stride=STRIDE, normalize=True)[0]
    a, b = standardize(ndvi, pan, stats)
    return unet_infer.detect_tree(a, b, width=256, height=256, stride=STRIDE, normalize=False)[0]


mode = sys.argv[1]
if mode == 'patches':
    df = pd.read_csv(sys.argv[2]); out = sys.argv[3]; os.makedirs(out, exist_ok=True)
    assert set(df.city) <= set(CITIES)
    res = {k: [] for k in ('P256', 'P', 'T', 'S')}; info = []
    for r in df.itertuples():
        t0 = time.time()
        px, py = (r.right - r.left) / 256, (r.top - r.bottom) / 256
        m = (CTX - 256) // 2
        ctx_tr = Affine(px, 0, r.left - m * px, 0, -py, r.top + m * py)
        ndvi, pan = read(r.mosaic, ctx_tr, CTX, CTX)
        c0 = m - (CROP - 256) // 2; c = (CROP - 256) // 2
        cn, cp = ndvi[c0:c0 + CROP, c0:c0 + CROP], pan[c0:c0 + CROP, c0:c0 + CROP]
        pn, pp = ndvi[m:m + 256, m:m + 256], pan[m:m + 256, m:m + 256]
        x = image_normalize(np.stack((pn, pp), -1), axis=(0, 1))
        res['P256'].append(np.squeeze(unet_infer.model.predict(x[None], verbose=0)))
        res['P'].append(run(cn, cp, 'P')[c:c + 256, c:c + 256])
        res['T'].append(run(cn, cp, 'T', valid_stats(ndvi, pan))[c:c + 256, c:c + 256])
        res['S'].append(run(cn, cp, 'S', scene_stats(r.mosaic))[c:c + 256, c:c + 256])
        info.append(dict(frame=r.frame, patch=r.patch, ctx_valid=float(((pan > 0) & (ndvi > 0)).mean()),
                         patch_valid=float(((pp > 0) & (pn > 0)).mean())))
        print('OK', r.patch, f'{time.time() - t0:.0f}s', flush=True)
    np.savez_compressed(f'{out}/patches_prob.npz', frames=df.frame.values, **{k: np.array(v, np.float16) for k, v in res.items()})
    pd.DataFrame(info).to_csv(f'{out}/patches_info.csv', index=False)
else:
    lines = [l.split() for l in open(sys.argv[2]) if l.strip()]
    first, k, out = int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    os.makedirs(f'{out}/tiles', exist_ok=True)
    for name, _, mosaic, L, T, DX, w, h in lines[first:first + k]:
        assert any(f'_{c}_' in name for c in CITIES)
        t0 = time.time()
        tr = Affine(float(DX), 0, float(L), 0, -float(DX), float(T)); w, h = int(w), int(h)
        ndvi, pan = read(mosaic, tr, w, h)
        prof = dict(driver='GTiff', width=w, height=h, count=1, dtype='uint8', crs='EPSG:4326', transform=tr, compress='deflate')
        for v, st in (('P', None), ('T', valid_stats(ndvi, pan)), ('S', scene_stats(mosaic))):
            p = run(ndvi, pan, v, st)
            with rasterio.open(f'{out}/tiles/{v}_{name}', 'w', **prof) as o:
                o.write(np.round(np.clip(p, 0, 1) * 255).astype(np.uint8), 1)
        print('OK', name, f'{time.time() - t0:.0f}s', flush=True)
