"""
07_artifact_screen.py -- screen large connected prediction regions for non-canopy artefacts
(solid, grid-aligned rectangular blocks such as no-data / scene-edge regions predicted as tree).

For every hole-filled 8-connected component >= MIN_PX pixels (the parent regions the split
algorithm worked on) compute:
  npx, bbox fill ratio,
  longest perfectly straight horizontal / vertical boundary run (pixels),
  share of the component inside fully-positive 101x101 windows (binary erosion),
and write diagnostics/large_components.csv plus a thumbnail per component for visual review.
A component is FLAGGED as a probable artefact when its longest straight grid-aligned edge is
>= EDGE_PX (natural canopy edges from the U-Net are not pixel-straight for 250 m) AND >= 50 %
of it lies in fully-positive 101x101 windows. Flags are proposals for visual confirmation only.
"""
import sys
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage as ndi
from multiprocessing import Pool
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

import os
ROOT = os.environ.get('URBAN_DATA', '/path/to/Urban_data')
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402
AUD = paths.AUDIT
DIAG = paths.DIAG
MIN_PX = 20000
EDGE_PX = 500
S8 = ndi.generate_binary_structure(2, 2)


def longest_run(b):
    """longest run of True along axis 1, over all rows."""
    best = 0
    for row in b:
        if not row.any():
            continue
        d = np.diff(np.concatenate([[0], row.view(np.int8), [0]]))
        s, e = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
        best = max(best, int((e - s).max()))
    return best


def screen(name):
    with rasterio.open(paths.pred_path(name)) as so:
        o = so.read(1) > 0
    fill = ndi.binary_fill_holes(o)
    lab, n = ndi.label(fill, S8)
    sizes = np.bincount(lab.ravel())
    big = np.nonzero(sizes >= MIN_PX)[0]
    big = big[big > 0]
    out = []
    if big.size == 0:
        return out
    objs = ndi.find_objects(lab)
    for k in big:
        sl = objs[k - 1]
        m = lab[sl] == k
        pad = np.pad(m, 1)
        top = pad[1:-1, 1:-1] & ~pad[:-2, 1:-1]
        bot = pad[1:-1, 1:-1] & ~pad[2:, 1:-1]
        lef = pad[1:-1, 1:-1] & ~pad[1:-1, :-2]
        rig = pad[1:-1, 1:-1] & ~pad[1:-1, 2:]
        h_edge = max(longest_run(top), longest_run(bot))
        v_edge = max(longest_run(lef.T), longest_run(rig.T))
        solid = ndi.binary_erosion(m, np.ones((101, 101), bool), border_value=0)
        solid = ndi.binary_dilation(solid, np.ones((101, 101), bool)) & m
        r = {'tile': name, 'parent_cc8': int(k), 'npx': int(sizes[k]),
             'bbox_h': m.shape[0], 'bbox_w': m.shape[1], 'bbox_fill': float(m.mean()),
             'max_straight_h_edge_px': h_edge, 'max_straight_v_edge_px': v_edge,
             'share_in_solid_101px_windows': float(solid.sum() / m.sum()),
             'row0': sl[0].start, 'col0': sl[1].start}
        r['flag_artifact'] = bool(max(h_edge, v_edge) >= EDGE_PX and r['share_in_solid_101px_windows'] >= 0.5)
        f = max(1, max(m.shape) // 400)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(o[sl][::f, ::f], cmap='Greens', interpolation='nearest')
        ax.contour(m[::f, ::f], [0.5], colors='r', linewidths=0.6)
        ax.set_title(f"{name[:-4]}\ncc{k} {sizes[k]:,} px edge {max(h_edge, v_edge)} solid {r['share_in_solid_101px_windows']:.2f}",
                     fontsize=6)
        ax.set_xticks([]); ax.set_yticks([])
        fig.savefig(f"{AUD}/figures/large_components/{name[:-4]}__cc{k}.png", dpi=90, bbox_inches='tight')
        plt.close(fig)
        out.append(r)
    return out


if __name__ == '__main__':
    import os
    os.makedirs(f'{AUD}/figures/large_components', exist_ok=True)
    scan = pd.read_csv(f'{DIAG}/tile_scan.csv')
    # only tiles whose largest split instance's parent may be large: use crown tables
    names = []
    for f in scan.filename:
        d = pd.read_parquet(f'{DIAG}/crowns/{f[:-4]}.split.parquet', columns=['parent_cc8_npx'])
        if len(d) and d.parent_cc8_npx.max() >= MIN_PX:
            names.append(f)
    print('tiles with components >=', MIN_PX, 'px:', len(names), flush=True)
    rows = []
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 6) as p:
        for out in p.imap_unordered(screen, names):
            rows.extend(out)
    df = pd.DataFrame(rows).sort_values('npx', ascending=False)
    df['city'] = df.tile.str.extract(r'_(austin|bloomington|cupertino|surrey)_')[0]
    df.to_csv(f'{DIAG}/large_components.csv', index=False)
    print(df.groupby('city').agg(n=('npx', 'size'), flagged=('flag_artifact', 'sum'), px=('npx', 'sum')))
