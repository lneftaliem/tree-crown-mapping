"""
16_scan_new_tiles.py -- incremental version of 01_tile_scan.py for the full (archive + regenerated) rebuild.

Runs 01_tile_scan.scan() unchanged on the tiles in SPLIT_DIR that have no row in an existing tile_scan.csv,
writes their crown tables to DIAG_DIR/crowns, and writes DIAG_DIR/tile_scan.csv = existing rows + new rows.
The 2,006 archive tiles are unchanged inputs, so their rows and crown tables (linked into DIAG_DIR/crowns)
are reused; a re-scan of an archive tile reproduced its stored tables exactly.
Usage: SPLIT_DIR=... PRED_DIR=... DIAG_DIR=... python 16_scan_new_tiles.py <existing tile_scan.csv> [n_workers]
"""
import os, sys, importlib.util
import pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402

spec = importlib.util.spec_from_file_location('tile_scan', os.path.join(os.path.dirname(os.path.abspath(__file__)), '01_tile_scan.py'))
ts = importlib.util.module_from_spec(spec); sys.modules['tile_scan'] = ts; spec.loader.exec_module(ts)  # registered so Pool can pickle scan()

if __name__ == '__main__':
    old = pd.read_csv(sys.argv[1])
    nw = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    names = sorted(f.replace('center_based_relabeled_', '') for f in os.listdir(paths.SPLIT_DIR)
                   if f.startswith('center_based_relabeled_') and f.endswith('.tif'))
    new = [n for n in names if n not in set(old.filename)]
    assert all(any(f'_{c}_' in n for c in ('austin', 'bloomington', 'cupertino', 'surrey')) for n in new)
    print(f'{len(names)} tiles, {len(old)} already scanned, {len(new)} new', flush=True)
    os.makedirs(ts.CROWN_DIR, exist_ok=True)
    with Pool(nw) as pool:
        rows = pool.map(ts.scan, new, chunksize=1)
    df = pd.concat([old, pd.DataFrame(rows)]).sort_values('filename')
    assert df.filename.is_unique and set(df.filename) == set(names)
    df.to_csv(f'{paths.DIAG}/tile_scan.csv', index=False)
    print('done', len(df))
