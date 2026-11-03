"""
sample_chips.py -- simple random sample of crowns of a dataset per city, rendered as PAN + NDVI chips (128 px = 64 m,
centred on the crown; the sampled crown outlined in magenta, other kept crowns in yellow) for visual land-cover
classification. Writes <out>/sample_<city>.csv and <out>/sheets/<city>_NN.png (16 crowns per sheet). Read-only on data.
Usage: python sample_chips.py <dataset> <out> <n_per_city> <seed>
"""
import sys, os
import numpy as np, pandas as pd, pyogrio
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chips

DS, OUT, N, SEED = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
os.makedirs(f'{OUT}/sheets', exist_ok=True)
H = 64
for ci, city in enumerate(chips.CITIES):
    p = pyogrio.read_dataframe(f'{DS}/{city}_ripley_points_with_income.geojson',
                               columns=['tree_id', 'source_tile', 'instance_label', 'crown_area_m2', 'block_group_id', 'median_household_income'])
    s = p.sample(N, random_state=SEED + ci).reset_index(drop=True)
    s['lon'], s['lat'] = s.geometry.x, s.geometry.y
    nd = pd.read_parquet(f'{chips.P}/work_v1b/diag/crown_ndvi_final_{city}.parquet')
    s = s.drop(columns='geometry')
    kept = p.groupby('source_tile').instance_label.apply(lambda x: set(x.astype(int)))
    rows = []
    for i, r in s.iterrows():
        g = chips.GEO.loc[r.source_tile]; px = (g.R - g.L) / g.o_w
        rr, cc = int((g['T'] - r.lat) / px), int((r.lon - g.L) / px)
        pan, ndv, lab, _ = chips.window(r.source_tile, rr - H, cc - H, 2 * H, 2 * H)
        rows.append((pan, ndv, lab, int(r.instance_label), kept.get(r.source_tile, set())))
    s['chip'] = [f'{city[:3].upper()}{i:03d}' for i in range(len(s))]
    s.to_csv(f'{OUT}/sample_{city}.csv', index=False)
    for k in range(0, len(s), 16):
        fig, ax = plt.subplots(4, 8, figsize=(24, 12.6))
        for j, a2 in enumerate(ax.reshape(-1, 2)):
            for a in a2: a.axis('off')
            if k + j >= len(s): continue
            pan, ndv, lab, L, kp = rows[k + j]
            tgt = find_boundaries(lab == L, mode='inner') & (lab == L)
            oth = np.isin(lab, list(kp)) & (lab != L) & (lab > 0)
            ob = find_boundaries(np.where(oth, lab, 0), mode='inner') & oth
            rgb = np.dstack([chips.stretch(pan)] * 3); rgb[ob] = (1, 0.85, 0); rgb[tgt] = (1, 0, 1)
            a2[0].imshow(rgb); a2[0].set_title(f'{s.chip[k+j]}  PAN  {s.crown_area_m2[k+j]:.0f} m²', fontsize=9)
            a2[1].imshow(ndv, cmap='RdYlGn', vmin=-0.1, vmax=0.8)
            t = np.zeros(ndv.shape + (4,)); t[tgt] = (1, 0, 1, 1); a2[1].imshow(t)
            a2[1].set_title('NDVI', fontsize=9)
        plt.tight_layout(); plt.savefig(f'{OUT}/sheets/{city}_{k//16:02d}.png', dpi=72); plt.close()
    print(city, 'done', flush=True)
