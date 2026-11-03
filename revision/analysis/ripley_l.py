"""
ripley_l.py -- Figure S6 and the Ripley's L statistics (Supplementary Note 5).

The original L-function code (run by a co-author) is not in the repository, the local project or the Sherlock account;
this is a re-implementation of the published specification (Supplementary Note 5):
  K(r) = A / N^2 * sum_i sum_{j != i} 1{d_ij <= r}        (no edge correction; A = area of the convex hull of the trees)
  L(r) = sqrt(K(r) / pi) - r,   r = 2, 4, ..., 50 m
  per census area (block group / dissemination area); areas with < 50 trees excluded; areas with > 100,000 trees
  subsampled to 100,000 (seed 0); for census areas made of several disconnected polygons, L(r) is computed per polygon
  and averaged weighted by polygon area (polygons with < 50 trees are skipped; if none qualifies the whole area is used);
  AUC = trapezoidal area under L(r) over 2-50 m; Spearman(AUC, median household income) over census areas with income;
  income quartiles (equal-frequency within city over those areas), L(r) averaged within quartile weighted by area.
Interpretation choices where the note is silent (documented in the manifest): census area = the municipality-clipped
unit geometry used everywhere else; 'weighted by area' uses that census-area area; trees are the dataset's crown
centroids in the city UTM zone, assigned to units by their block_group_id attribute and to polygon parts by location.
Writes analysis_output/ripley/{ripley_L_by_unit.csv, ripley_auc_income_spearman.csv, ripley_L_by_quartile.csv,
FigS6_ripley_L_by_income_quartile.{png,pdf,svg}}. The figure follows the published layout (2 x 2, shared y axis,
matplotlib default colours, legend 'quartile').
"""
import os, sys
import numpy as np, pandas as pd, geopandas as gpd, pyogrio, shapely
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'repo_adapted'))
import repro_data  # noqa: E402

OUT = os.path.join(repro_data.BASE_DIR, 'analysis_output', 'ripley')
os.makedirs(OUT, exist_ok=True)
CITIES = ['austin', 'bloomington', 'cupertino', 'surrey']
UTM = {'austin': 32614, 'bloomington': 32616, 'cupertino': 32610, 'surrey': 32610}
R = np.arange(2, 51, 2, dtype=float)
MIN_TREES, MAX_TREES = 50, 100_000
rng = np.random.default_rng(0)


def l_function(xy):
    if len(xy) > MAX_TREES:
        xy = xy[rng.choice(len(xy), MAX_TREES, replace=False)]
    n = len(xy)
    A = shapely.convex_hull(shapely.multipoints(xy)).area
    t = cKDTree(xy)
    pairs = t.count_neighbors(t, R).astype(float) - n  # ordered pairs i != j with d <= r
    K = A / n ** 2 * pairs
    return np.sqrt(K / np.pi) - R


rows = []
for city in CITIES:
    src = repro_data.ensure_city_points(city)
    p = pyogrio.read_dataframe(src, columns=['block_group_id', 'median_household_income', 'longitude', 'latitude'],
                               read_geometry=False)
    import pyproj
    x, y = pyproj.Transformer.from_crs(4326, UTM[city], always_xy=True).transform(p.longitude.values, p.latitude.values)
    p['x'], p['y'] = x, y
    p['median_household_income'] = pd.to_numeric(p.median_household_income, errors='coerce')
    repro_data.ensure_boundaries(city)
    bg = gpd.read_file(os.path.join(repro_data.BOUNDARIES_STAGE_DIR, f'{city}_exact_block_groups.geojson')).to_crs(UTM[city])
    bg['uid'] = bg['DAUID' if city == 'surrey' else 'GEOID'].astype('int64')
    geom = bg.set_index('uid').geometry
    for uid, g in p.dropna(subset=['block_group_id']).groupby(p.block_group_id.dropna().astype('int64')):
        if len(g) < MIN_TREES or uid not in geom.index:
            continue
        xy = g[['x', 'y']].values
        parts = list(geom[uid].geoms) if geom[uid].geom_type == 'MultiPolygon' else [geom[uid]]
        Ls, ws = [], []
        if len(parts) > 1:
            for part in parts:
                m = shapely.contains_xy(part, xy[:, 0], xy[:, 1])
                if m.sum() >= MIN_TREES:
                    Ls.append(l_function(xy[m])); ws.append(part.area)
        if not Ls:
            Ls, ws = [l_function(xy)], [1.0]
        L = np.average(np.vstack(Ls), axis=0, weights=ws)
        rows.append(dict(city=city, unit=uid, n_trees=len(g), n_polygons_used=len(Ls), unit_area_m2=geom[uid].area,
                         median_income=g.median_household_income.iloc[0], auc=np.trapezoid(L, R),
                         **{f'L_{int(r)}': v for r, v in zip(R, L)}))
    print(city, sum(r['city'] == city for r in rows), 'census areas', flush=True)
D = pd.DataFrame(rows)
D.to_csv(f'{OUT}/ripley_L_by_unit.csv', index=False)

sp, qrows = [], []
fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharey=True)
Lcols = [f'L_{int(r)}' for r in R]
for ax, city in zip(axes.flat, CITIES):
    d = D[(D.city == city) & D.median_income.notna()].copy()
    s = spearmanr(d.auc, d.median_income)
    sp.append(dict(city=city, n_census_areas=len(d), spearman_rho_auc_income=s.statistic, p_value=s.pvalue,
                   share_units_clustered_all_r=float((d[Lcols] > 0).all(axis=1).mean())))
    d['quartile'] = pd.qcut(d.median_income, 4, labels=[1, 2, 3, 4]).astype(int)
    for q, gq in d.groupby('quartile'):
        Lq = np.average(gq[Lcols].values, axis=0, weights=gq.unit_area_m2.values)
        qrows.append(dict(city=city, quartile=q, n=len(gq), **dict(zip(Lcols, Lq))))
        ax.plot(R, Lq, label=str(q))
    ax.set_title(city.title(), fontsize=20, fontweight='bold')
    ax.grid(True)
    ax.tick_params(labelsize=8)
    ax.set_ylim(bottom=0)
for ax in axes[1]:
    ax.set_xlabel('$r$')
for ax in axes[:, 0]:
    ax.set_ylabel(r'$\widehat{L}(r)$')
axes[0, 0].legend(title='quartile', fontsize=8)
for ext in ('png', 'pdf', 'svg'):
    fig.savefig(f'{OUT}/FigS6_ripley_L_by_income_quartile.{ext}', dpi=300, bbox_inches='tight')
pd.DataFrame(sp).to_csv(f'{OUT}/ripley_auc_income_spearman.csv', index=False)
pd.DataFrame(qrows).to_csv(f'{OUT}/ripley_L_by_quartile.csv', index=False)
print(pd.DataFrame(sp).round(4).to_string(index=False))
