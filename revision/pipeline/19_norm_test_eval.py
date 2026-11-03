"""
19_norm_test_eval.py -- evaluate the inference-normalization test (code/sherlock/norm_test.py) at the chosen threshold 0.7.

Patches (50 re-extractable validation patches): pixel accuracy / precision / recall / Dice / IoU against the annotation masks,
  connected-component object count error, NDVI of false-positive pixels; per variant P256, P, T, S. P256 is also compared
  with the threshold-selection predictions (threshold_v1_20261101/bin_0.7) as a re-extraction check.
Tiles (4 Figure 6 non-vegetated tiles + 8 random v1 tiles): positive share, share of positive pixels with NDVI < 0.2,
  connected components >= 3 m2 and the share of them with mean NDVI < 0.2; P is also compared with the v1 probabilities.
Writes <test_dir>/norm_test_patches.csv, norm_test_tiles.csv, norm_test_fig6_windows.png.
Usage: python 19_norm_test_eval.py <test_dir> (contains patches.csv, tile_roles.csv, tiles.txt, patches_prob.npz, tiles/)
"""
import os, sys, glob
import numpy as np, pandas as pd, rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from affine import Affine
from scipy import ndimage
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

U = '/path/to/Urban_data'
PGC = f'{U}/NASA:PGC/Training_06062024'
P = f'{U}/tree_crowns_v1_20261101'
TH = 0.7
d = sys.argv[1]
VAR = ['P256', 'P', 'T', 'S']


def mosaic_path(city, m):
    return f"{PGC}/{city}/{'_'.join(m.split('_')[-3:-1])}/{m}"


def ndvi_on(city, m, tr, w, h):
    with rasterio.open(mosaic_path(city, m)) as s, \
            WarpedVRT(s, crs='EPSG:4326', transform=tr, width=w, height=h, resampling=Resampling.nearest) as v:
        raw = v.read(2).astype(np.float64)
    return np.where(raw > 0, raw / 1000 - 1, np.nan)


def m2_per_px(tr):
    lat = tr.f + tr.e * 128
    return abs(tr.a) * 111320 * np.cos(np.radians(lat)) * abs(tr.e) * 110574


# ---------------- patches
pt = pd.read_csv(f'{d}/patches.csv')
z = np.load(f'{d}/patches_prob.npz'); assert (z['frames'] == pt.frame.values).all()
rows, tot = [], {v: np.zeros(4) for v in VAR}
fp_ndvi = {v: [] for v in VAR}
for i, r in enumerate(pt.itertuples()):
    with rasterio.open(f'{U}/validation_annotations_20261101/{r.patch.replace("ndvi_", "annotation_")}') as a:
        A = a.read(); A = (A.max(0) if A.ndim == 3 else A)[:256, :256] > 0
    with rasterio.open(f'{U}/threshold_v1_20261101/bin_0.7/pred_patch_{r.frame:03d}.tif') as p:
        B0 = p.read(1)[:256, :256] > 0
    tr = Affine((r.right - r.left) / 256, 0, r.left, 0, -(r.top - r.bottom) / 256, r.top)
    nd = ndvi_on(r.city, r.mosaic, tr, 256, 256)
    a_px = m2_per_px(tr)
    n_ref = int(sum((s * a_px >= 3) for s in np.bincount(ndimage.label(A)[0].ravel())[1:]))
    rec = dict(patch=r.patch, city=r.city, ref_objects=n_ref)
    for v in VAR:
        M = z[v][i].astype(np.float32) >= TH
        tp, fp, fn = (M & A).sum(), (M & ~A).sum(), (~M & A).sum(); tn = M.size - tp - fp - fn
        tot[v] += [tp, fp, fn, tn]
        fp_ndvi[v].append(nd[M & ~A])
        lab, k = ndimage.label(M)
        rec[f'{v}_objects'] = int(sum((s * a_px >= 3) for s in np.bincount(lab.ravel())[1:]))
        rec[f'{v}_dice'] = 2 * tp / max(2 * tp + fp + fn, 1)
        rec[f'{v}_pos_share'] = M.mean()
    rec['P256_vs_threshold_run_agreement'] = ((z['P256'][i].astype(np.float32) >= TH) == B0).mean()
    rows.append(rec)
pdf = pd.DataFrame(rows); pdf.to_csv(f'{d}/norm_test_patches.csv', index=False)
summ = []
for v in VAR:
    tp, fp, fn, tn = tot[v]
    f = np.concatenate(fp_ndvi[v]); f = f[~np.isnan(f)]
    summ.append(dict(variant=v, accuracy=(tp + tn) / tot[v].sum(), precision=tp / (tp + fp), recall=tp / (tp + fn),
                     dice=2 * tp / (2 * tp + fp + fn), iou=tp / (tp + fp + fn),
                     objects_pred=pdf[f'{v}_objects'].sum(), objects_ref=pdf.ref_objects.sum(),
                     mean_abs_count_err=(pdf[f'{v}_objects'] - pdf.ref_objects).abs().mean(),
                     fp_px_ndvi_median=np.median(f) if len(f) else np.nan, fp_px_share_ndvi_lt02=(f < 0.2).mean() if len(f) else np.nan))
summ = pd.DataFrame(summ); summ.to_csv(f'{d}/norm_test_patches_summary.csv', index=False)
print(summ.round(3).to_string(index=False))
print('P256 vs threshold-run binary agreement: median', pdf.P256_vs_threshold_run_agreement.median().round(4),
      'min', pdf.P256_vs_threshold_run_agreement.min().round(4))
print(pdf.groupby('city')[[f'{v}_dice' for v in VAR]].mean().round(3))

# ---------------- tiles
roles = pd.read_csv(f'{d}/tile_roles.csv').set_index('filename').role
lines = {l.split()[0]: l.split() for l in open(f'{d}/tiles.txt') if l.strip()}
trows, panels = [], {}
fig6 = pd.read_csv(f'{P}/outputs/runs/v1_final_fig6/Fig6_windows.csv').set_index('tile')
for name, (fn, _, mosaic, L, T, DX, w, h) in lines.items():
    city = next(c for c in ('austin', 'bloomington', 'cupertino', 'surrey') if f'_{c}_' in name)
    tr = Affine(float(DX), 0, float(L), 0, -float(DX), float(T)); w, h = int(w), int(h)
    nd = ndvi_on(city, mosaic, tr, w, h); a_px = m2_per_px(tr)
    with rasterio.open(f'{U}/v1_ndvi_pan_20261101/prob/{name}') as s:
        v1 = s.read(1)
    rec = dict(tile=name, city=city, role=roles[name], valid_share=np.isfinite(nd).mean())
    for v in ('P', 'T', 'S'):
        with rasterio.open(f'{d}/tiles/{v}_{name}') as s:
            pr = s.read(1)
        M = pr >= round(TH * 255)
        if v == 'P':
            rec['P_vs_v1_binary_agreement'] = (M == (v1 >= round(TH * 255))).mean()
        lab, k = ndimage.label(M)
        npx = np.bincount(lab.ravel()); ok = np.isfinite(nd)
        s_nd = np.bincount(lab[ok].ravel(), weights=nd[ok], minlength=k + 1); n_ok = np.bincount(lab[ok].ravel(), minlength=k + 1)
        keep = np.nonzero((npx * a_px >= 3) & (n_ok > 0))[0]; keep = keep[keep > 0]
        cm = s_nd[keep] / n_ok[keep]
        rec.update({f'{v}_pos_share': M.mean(), f'{v}_pos_ndvi_lt02': np.nanmean(nd[M] < 0.2) if M.any() else np.nan,
                    f'{v}_components_ge3m2': len(keep), f'{v}_components_ndvi_lt02': (cm < 0.2).mean() if len(cm) else np.nan,
                    f'{v}_veg_px_detected': (M & (nd >= 0.4)).sum() / max((nd >= 0.4).sum(), 1)})
        if name in fig6.index:
            rr, cc = int(fig6.loc[name, 'row']), int(fig6.loc[name, 'col'])
            panels.setdefault(name, {})[v] = M[rr:rr + 256, cc:cc + 256]
            panels[name]['ndvi'] = nd[rr:rr + 256, cc:cc + 256]
    trows.append(rec)
tdf = pd.DataFrame(trows); tdf.to_csv(f'{d}/norm_test_tiles.csv', index=False)
pd.set_option('display.width', 250)
print(tdf.round(3).to_string(index=False))

fig, ax = plt.subplots(len(panels), 4, figsize=(12, 3.1 * len(panels)))
for i, (name, p) in enumerate(panels.items()):
    ax[i, 0].imshow(p['ndvi'], cmap='RdYlGn', vmin=-0.2, vmax=0.8); ax[i, 0].set_ylabel(name.split('_')[3], fontsize=11)
    for j, v in enumerate(('P', 'T', 'S'), 1):
        ax[i, j].imshow(p['ndvi'], cmap='gray', vmin=-0.2, vmax=0.8); ax[i, j].imshow(np.ma.masked_where(~p[v], p[v]), cmap='autumn', alpha=0.6)
    for a_ in ax[i]:
        a_.set_xticks([]); a_.set_yticks([])
for j, t in enumerate(['NDVI (-0.2 to 0.8)', 'per patch (v1)', 'per tile', 'per scene']):
    ax[0, j].set_title(t)
fig.tight_layout(); fig.savefig(f'{d}/norm_test_fig6_windows.png', dpi=110)
print('written', d)
