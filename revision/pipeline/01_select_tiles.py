"""
01_select_tiles.py -- one authoritative, consistently processed prediction per geographic window (v1 dataset).

Inventory: every full (non-patch) prediction tile of the four cities known to the full build
(split_audit_20261031/diagnostics_full/georef_table.csv: 2,006 archived + 145 regenerated), the 6 regenerated
surrey_185_028 alias duplicates (regen_20261031, excluded in the full build), and the current-pipeline reruns of every
ARCHIVED in-city window (vintage_compare_20261101: 3 mixed scenes, all windows; full_rerun_20261101: the other 20 scenes,
in-city windows).  Reruns used the validated pipeline: verbatim full-scene EPSG:4326 reprojection, PAN in both U-Net
inputs, detect_tree 256 px / stride 224, threshold 0.7, model trees_20260120-1928_..._final.keras, separation -cn 8 -m 15.

Window = identical footprint (corner and size within 1e-3 px). Files with the same footprint under different names
(pan/ndvi, and the surrey_182_028 relabel of surrey_185_028) are the same window.
Compatibility of an archived file = pixel IoU between its binary mask and the current-pipeline rerun of the same window
on identical imagery (an empirical test of model weights, input channels, normalisation, reprojection/alignment, threshold
and windowing together); compatible when IoU >= IOU_OK.
Decision per in-city window, in this order:
  1. a regenerated file (current pipeline by construction)                      -> 'regenerated'
  2. else the compatible archived file with the highest IoU (ties: name order) -> 'archived_compatible'
  3. else the current-pipeline rerun                                            -> 'rerun_replaces_incompatible'
     (includes archived tiles that were empty although imagery is valid, e.g. surrey_186_028)
Every other file is recorded with its reason (redundant same-window file, incompatible vintage superseded, outside city,
excluded alias duplicate).  Windows outside the municipality (footprint share 0) contribute no crowns and are not used.
Nothing is copied or deleted: the output is a decision table plus symlink folders for the build.
Usage: python 01_select_tiles.py <out_dir>
"""
import os, sys, json, glob
import numpy as np, pandas as pd

ROOT = '/path/to/Urban_data'
OUT = sys.argv[1]
FULL = f'{ROOT}/split_audit_20261031/diagnostics_full'
ARCH_SPLIT = f'{ROOT}/split'
REGEN = f'{ROOT}/regen_20261031'
RERUNS = [f'{ROOT}/vintage_compare_20261101', f'{ROOT}/full_rerun_20261101']
CITIES = ('austin', 'bloomington', 'cupertino', 'surrey')
IOU_OK = 0.99

g = pd.read_csv(f'{FULL}/georef_table.csv')
g = g[g.city.isin(CITIES) & ~g.filename.str.contains('patch')].copy()
audit = pd.read_csv(f'{FULL}/empty_tile_audit.csv').set_index('filename')
g['city_share'] = g.filename.map(audit.city_share)
regen_files = set(os.listdir(f'{REGEN}/predictions'))
g['src'] = np.where(g.filename.isin(regen_files), 'regen', 'archive')
alias = pd.read_csv(f'{FULL}/excluded_alias_duplicates.csv')
assert g.city_share.notna().all()

# rerun statistics (vintage_compare.py JSONL): per archived file, IoU vs the rerun of the same window
st = pd.DataFrame([json.loads(l) for d in RERUNS for f in glob.glob(f'{d}/stats/*.jsonl') for l in open(f)])
assert 'error' not in st or st.error.isna().all(), st[st.error.notna()]
st = st.drop_duplicates('filename', keep='last')
st['iou'] = np.where(st.union_px > 0, st.inter_px / st.union_px.where(st.union_px > 0), 1.0)
rr = {}  # filename -> (rerun dir, rerun prediction file of its window)
for d in RERUNS:
    have = set(os.listdir(f'{d}/predictions'))
    for f in glob.glob(f'{d}/stats/*.jsonl'):
        s = pd.DataFrame([json.loads(l) for l in open(f)])
        s['k'] = s.col.round().astype(int).astype(str) + '_' + s.row.round().astype(int).astype(str) + '_' + s.w.astype(str)
        for k, grp in s.groupby('k'):
            cur = next(x for x in grp.filename if x in have)
            for x in grp.filename:
                rr[x] = (d, cur)

# window key: identical footprint within 1e-3 px
px = (g.R - g.L) / g.o_w
for c in ('L', 'T'):
    g[f'{c}q'] = np.round(g[c] / (px * 1e-3)).astype('int64')
g['window'] = g.city + '_' + g.Lq.astype(str) + '_' + g.Tq.astype(str) + '_' + g.o_w.astype(str) + 'x' + g.o_h.astype(str)
g = g.merge(st[['filename', 'iou', 'ex_px', 'cur_px', 'valid_px']], on='filename', how='left')
g['in_city'] = g.city_share > 0

rows = []
for w, grp in g.groupby('window'):
    grp = grp.sort_values('filename')
    base = dict(window=w, city=grp.city.iloc[0], L=grp.L.iloc[0], T=grp['T'].iloc[0], R=grp.R.iloc[0], B=grp.B.iloc[0],
                n_files=len(grp), files=';'.join(grp.filename))
    if not grp.in_city.iloc[0]:
        for r in grp.itertuples():
            rows.append(dict(base, filename=r.filename, src=r.src, iou=r.iou, decision='not_used_outside_city',
                             reason='footprint does not intersect the municipality'))
        continue
    reg = grp[grp.src == 'regen']
    arc = grp[grp.src == 'archive']
    assert arc.iou.notna().all(), f'archived in-city file without rerun: {list(arc.filename[arc.iou.isna()])}'
    if len(reg):
        sel, how = reg.filename.iloc[0], 'regenerated'
    elif (arc.iou >= IOU_OK).any():
        c = arc[arc.iou >= IOU_OK].sort_values(['iou', 'filename'], ascending=[False, True])
        sel, how = c.filename.iloc[0], 'archived_compatible'
    else:
        sel, how = arc.filename.iloc[0], 'rerun_replaces_incompatible'
    for r in grp.itertuples():
        if r.filename == sel:
            d, reason = how, {'regenerated': 'current pipeline (regen_20261031)',
                              'archived_compatible': f'archived prediction matches current pipeline (IoU {r.iou:.4f})',
                              'rerun_replaces_incompatible': (f'archived prediction incompatible (IoU {r.iou:.4f}); '
                                                              'current-pipeline rerun of the same window used')}[how]
        elif r.src == 'archive' and (r.iou < IOU_OK):
            d, reason = 'superseded_incompatible_vintage', f'IoU {r.iou:.4f} vs current pipeline; window represented by {sel}'
        else:
            d, reason = 'redundant_same_window', f'same footprint as selected {sel}'
        rows.append(dict(base, filename=r.filename, src=r.src, iou=r.iou, decision=d, reason=reason,
                         selected_for_window=sel, selected_how=how))
for r in alias.itertuples():  # regenerated alias duplicates (not in the full build's tables)
    rows.append(dict(filename=r.regen_tile, src='regen', decision='redundant_same_window',
                     reason=f'regenerated duplicate of archived {r.archived_tile_same_window} (alias 182_028/185_028)'))
inv = pd.DataFrame(rows)
assert inv.filename.is_unique

# content source for each selected window
sel = inv[inv.decision.isin(['regenerated', 'archived_compatible', 'rerun_replaces_incompatible'])].copy()


def sources(r):
    f = r.filename
    if r.decision == 'regenerated':
        return f'{REGEN}/split/center_based_relabeled_{f}', f'{REGEN}/predictions/{f}'
    if r.decision == 'archived_compatible':
        return f'{ARCH_SPLIT}/center_based_relabeled_{f}', f'zip:{f}'
    d, cur = rr[f]
    return f'{d}/split/center_based_relabeled_{cur}', f'{d}/predictions/{cur}'


sel[['split_source', 'pred_source']] = sel.apply(lambda r: pd.Series(sources(r)), axis=1)
for p in sel.split_source:
    assert os.path.exists(p), p
os.makedirs(f'{OUT}/split_links'); os.makedirs(f'{OUT}/pred_links')
for r in sel.itertuples():
    os.symlink(r.split_source, f'{OUT}/split_links/center_based_relabeled_{r.filename}')
    if not r.pred_source.startswith('zip:'):  # archived predictions are read from used_predictions.zip by paths.pred_path
        os.symlink(r.pred_source, f'{OUT}/pred_links/{r.filename}')
inv = inv.merge(sel[['filename', 'split_source', 'pred_source']], on='filename', how='left')
inv.to_csv(f'{OUT}/tile_inventory.csv', index=False)

pd.set_option('display.width', 200)
print(pd.crosstab([inv.city, inv.src], inv.decision, margins=True))
w = inv[inv.decision.isin(['regenerated', 'archived_compatible', 'rerun_replaces_incompatible'])]
print('in-city windows', w.window.nunique(), '| selected files', len(w), '| one per window:', w.window.is_unique)
print(inv[inv.src == 'archive'].groupby('city').iou.describe(percentiles=[.01, .05, .5]).round(4))
