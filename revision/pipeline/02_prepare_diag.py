"""
02_prepare_diag.py -- diagnostics folder for the v1 build, reusing validated per-tile scans where the tile is unchanged.

For every SELECTED tile (01_select_tiles.py) whose content is the same raster pair as in the full build (archived
compatible: Christian's split raster + archived prediction; regenerated: regen split + prediction), the full build's
tile_scan row and crown tables are reused (links). Tiles whose content is a current-pipeline rerun are left out of
tile_scan_base.csv so 16_scan_new_tiles.py scans them. artifact_components.csv keeps only entries of selected tiles
whose content is unchanged (artefact components were identified on the old predictions; rerun tiles are screened anew).
Usage: DIAG_DIR=... python 02_prepare_diag.py <tile_inventory.csv>
"""
import os, sys
import pandas as pd

FULL = '/path/to/Urban_data/split_audit_20261031/diagnostics_full'
DIAG = os.environ['DIAG_DIR']
inv = pd.read_csv(sys.argv[1])
sel = inv[inv.decision.isin(['regenerated', 'archived_compatible', 'rerun_replaces_incompatible'])]
same = set(sel.filename[sel.decision != 'rerun_replaces_incompatible'])
ts = pd.read_csv(f'{FULL}/tile_scan.csv')
assert same <= set(ts.filename), sorted(same - set(ts.filename))[:5]
ts[ts.filename.isin(same)].to_csv(f'{DIAG}/tile_scan_base.csv', index=False)
os.makedirs(f'{DIAG}/crowns', exist_ok=True)
n = 0
for f in os.listdir(f'{FULL}/crowns'):
    if f.rsplit('.', 2)[0] + '.tif' in same:
        dst = f'{DIAG}/crowns/{f}'
        if not os.path.lexists(dst):
            os.symlink(os.path.realpath(f'{FULL}/crowns/{f}'), dst); n += 1
art = pd.read_csv(f'{FULL}/artifact_components.csv')
keep = art[art.tile.isin(same)]
keep.to_csv(f'{DIAG}/artifact_components.csv', index=False)
pd.DataFrame({'tile': sorted(set(art.tile) - set(keep.tile))}).to_csv(f'{DIAG}/artifact_components_dropped_tiles.csv', index=False)
print(f'reused scans: {len(same)} tiles ({n} crown tables linked); to scan: {len(sel) - len(same)}; '
      f'artefact components kept {len(keep)} of {len(art)}')
