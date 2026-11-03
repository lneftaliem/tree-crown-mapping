"""
regen_v1b.py -- v1b citywide inference (v1 with the input channels in TRAINING order), one scene: NDVI + PAN inputs (the intended model design and the input
configuration of the validation), every in-city window of the scene. Four cities only.

Identical to the validated regeneration pipeline (regen_tile.py / vintage_compare.py) except for the input channels:
  verbatim full-scene EPSG:4326 reprojection of the 2-band PV mosaic (reproject_verbatim.reproject_to_4326);
  detect_tree(ndvi_band = band 2, pan_band = band 1), 256 x 256 windows, stride 224, per-patch per-channel
  image_normalize (unet_infer.detect_tree, verbatim from 3-RasterAnalysis); model trees_20260120-1928_..._final.keras;
  binary mask = probability >= THRESH (env, chosen on the 113 validation patches).
Writes $OUT/prob/<name> (uint8, round(255 * p), deflate) and $OUT/predictions/<name> (uint8 0/1, deflate) on the window's
grid with CRS and transform; one JSON line per window in $OUT/stats/<scene>.jsonl.
v1b CHANGE (2026-11-01): the model was trained and validated with channel 0 = mosaic band 1 (Panchromatic) and channel 1 =
band 2 (NDVI): 1-Preprocessing writes band index 0 to the files named ndvi_*.png and band index 1 to pan_*.png (bands = [0, 1];
verified: the validation ndvi_*.png correlate r = 0.999 with band 1, pan_*.png r = 1.000 with band 2), and the validation stacks
[ndvi_*.png, pan_*.png]. v1 called detect_tree(ndvi_band = band 2, pan_band = band 1), i.e. swapped. v1b passes band 1 as the
first argument (detect_tree's `ndvi_band` = channel 0) and band 2 as the second. Nothing else differs.
Usage: THRESH=0.7 python regen_v1b.py <scene_manifest> <outdir> <scratch_dir>
  manifest lines: <filename> <src> <mosaic> <L> <T> <dx> <w> <h>
"""
import os, sys, json, time
import numpy as np, rasterio
from rasterio import windows

chunk, outdir, scratch = sys.argv[1:4]
RAW = os.environ['RAW_DIR']
THRESH = float(os.environ['THRESH'])
CITIES = ('austin', 'bloomington', 'cupertino', 'surrey')
STRIDE = 224
rows = [l.split() for l in open(chunk) if l.strip()]
mosaics = {r[2] for r in rows}
assert len(mosaics) == 1, 'one scene per manifest'
mosaic = mosaics.pop()
assert any(mosaic.startswith(f'LN_{c}_') for c in CITIES) and all(any(f'_{c}_' in r[0] for c in CITIES) for r in rows)
for d in ('prob', 'predictions', 'stats'):
    os.makedirs(f'{outdir}/{d}', exist_ok=True)
stem = mosaic.replace('_mosaic.tif', '')

from reproject_verbatim import reproject_to_4326
reproj = f'{scratch}/{stem}_mosaic_reprojected.tif'
t0 = time.time(); reproject_to_4326(f'{RAW}/{mosaic}', reproj); print(f'reprojected in {time.time() - t0:.0f}s', flush=True)

import tensorflow as tf
n = int(os.environ.get('SLURM_CPUS_PER_TASK', '4'))
tf.config.threading.set_intra_op_parallelism_threads(n); tf.config.threading.set_inter_op_parallelism_threads(2)
sys.path.insert(0, os.environ['NB_DIR'])
from core.losses import tversky, accuracy, dice_coef, dice_loss, specificity, sensitivity
from tensorflow.keras.models import load_model
import unet_infer
unet_infer.model = load_model(os.environ['MODEL'], custom_objects={'tversky': tversky, 'dice_coef': dice_coef,
                              'dice_loss': dice_loss, 'accuracy': accuracy, 'specificity': specificity,
                              'sensitivity': sensitivity}, compile=False)

out = open(f'{outdir}/stats/{stem}.jsonl', 'a')
with rasterio.open(reproj) as R:
    T0 = R.transform
    for name, src, _, L, T, DX, w, h in rows:
        L, T, DX, w, h = float(L), float(T), float(DX), int(w), int(h)
        t0 = time.time()
        c, r = (L - T0.c) / T0.a, (T0.f - T) / -T0.e
        rec = dict(filename=name, scene=stem, col=c, row=r, w=w, h=h, thresh=THRESH, inputs='ch0_band1_pan+ch1_band2_ndvi')
        try:
            if os.path.exists(f'{outdir}/predictions/{name}'):
                print('SKIP', name, flush=True); continue
            assert abs(T0.a - DX) / DX < 1e-6, 'pixel size differs from scene'
            assert abs(c - round(c)) < 1e-3 and abs(r - round(r)) < 1e-3, 'tile not on scene grid'
            win = windows.Window(int(round(c)), int(round(r)), w, h)
            assert win.col_off >= 0 and win.row_off >= 0 and win.col_off + w <= R.width and win.row_off + h <= R.height
            pan = R.read(1, window=win).astype(np.float32)
            ndvi = R.read(2, window=win).astype(np.float32)
            m, _ = unet_infer.detect_tree(pan, ndvi, width=256, height=256, stride=STRIDE, normalize=True)
            prof = dict(driver='GTiff', width=w, height=h, count=1, dtype='uint8', crs=R.crs,
                        transform=windows.transform(win, T0), compress='deflate')
            for sub, arr in (('prob', np.round(np.clip(m, 0, 1) * 255)), ('predictions', m >= THRESH)):
                tmp = f'{outdir}/{sub}/{name}.partial'
                with rasterio.open(tmp, 'w', **prof) as o:
                    o.write(arr.astype(np.uint8), 1)
                os.replace(tmp, f'{outdir}/{sub}/{name}')
            valid = pan > 0
            rec.update(valid_px=int(valid.sum()), pos_px=int((m >= THRESH).sum()), pos_px_valid=int(((m >= THRESH) & valid).sum()),
                       secs=round(time.time() - t0))
            print('OK', name, rec['pos_px'], f"{rec['secs']}s", flush=True)
        except Exception as e:
            rec['error'] = f'{type(e).__name__}: {e}'; print('FAIL', name, e, flush=True)
        out.write(json.dumps(rec) + '\n'); out.flush()
os.remove(reproj)
