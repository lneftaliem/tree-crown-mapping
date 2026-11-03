"""
ripley_discrepancy.py -- why the documented-method L(r) (ripley_l.py) differs from the published Figure S6 levels.

Run on the ORIGINAL dataset (explicitly labelled comparison run). For every census area that ripley_l.py uses, computes
K(r) at r = 2 and 50 m under alternative conventions, each changing ONE element of the documented specification:
  doc            ordered pairs i != j, / N^2, A = convex hull of the trees (per polygon part, area-weighted)  [primary]
  n_nm1          / N(N-1) instead of / N^2
  unordered      unordered pairs (each pair once)
  pairs_over_half  ordered pairs / (N(N-1)/2)
  census_area    A = census-polygon area instead of the convex hull
  whole_unit     multi-part units treated as one point set (no per-part average)
and compares the area-weighted quartile means of L(r) with values read from the published Figure S6 (digitised by eye
from the manuscript image, +-1 unit; extracted figure: manuscript_original/figures_extracted/SupplementaryInformation/
image7.png). Also recomputes the AUC-income Spearman test and an alternative quartile test (Kruskal-Wallis of AUC across
quartiles) for each convention -- reported side by side, the documented Spearman test remains the primary result.
Writes analysis_output/ripley/discrepancy_{levels,tests}.csv.
"""
import os, sys
import numpy as np, pandas as pd, geopandas as gpd, pyogrio, shapely, pyproj
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, kruskal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'repo_adapted'))
import repro_data  # noqa: E402

OUT = os.path.join(repro_data.BASE_DIR, 'analysis_output', 'ripley')
os.makedirs(OUT, exist_ok=True)
UTM = {'austin': 32614, 'bloomington': 32616, 'cupertino': 32610, 'surrey': 32610}
R = np.arange(2, 51, 2, dtype=float)
rng = np.random.default_rng(0)
# published Figure S6, quartile curves read at r = 2 and r = 50 (approximate)
PUB = {('austin', 1): (15, 74), ('austin', 2): (13, 64), ('austin', 3): (15, 83), ('austin', 4): (13, 74),
       ('bloomington', 1): (14, 35), ('bloomington', 2): (14, 54), ('bloomington', 3): (13, 53), ('bloomington', 4): (8, 54),
       ('cupertino', 1): (10, 23), ('cupertino', 2): (10, 35), ('cupertino', 3): (9, 30), ('cupertino', 4): (9, 30),
       ('surrey', 1): (5, 35), ('surrey', 2): (5, 36), ('surrey', 3): (4, 30), ('surrey', 4): (5, 60)}
VARIANTS = ['doc', 'n_nm1', 'unordered', 'pairs_over_half', 'census_area', 'whole_unit']


def counts(xy):
    if len(xy) > 100_000:
        xy = xy[rng.choice(len(xy), 100_000, replace=False)]
    t = cKDTree(xy)
    return len(xy), t.count_neighbors(t, R).astype(float) - len(xy), shapely.convex_hull(shapely.multipoints(xy)).area


def L_of(K):
    return np.sqrt(np.maximum(K, 0) / np.pi) - R


rows = []
for city in UTM:
    p = pyogrio.read_dataframe(repro_data.ensure_city_points(city), columns=['block_group_id', 'median_household_income',
                               'longitude', 'latitude'], read_geometry=False)
    x, y = pyproj.Transformer.from_crs(4326, UTM[city], always_xy=True).transform(p.longitude.values, p.latitude.values)
    p['x'], p['y'] = x, y
    p['inc'] = pd.to_numeric(p.median_household_income, errors='coerce')
    bfile, bgfile = repro_data.ensure_boundaries(city)
    bg = gpd.read_file(bgfile).to_crs(UTM[city]); bg['uid'] = bg['DAUID' if city == 'surrey' else 'GEOID'].astype('int64')
    geom = bg.set_index('uid').geometry
    for uid, g in p.dropna(subset=['block_group_id']).groupby(p.block_group_id.dropna().astype('int64')):
        if len(g) < 50 or uid not in geom.index:
            continue
        xy = g[['x', 'y']].values
        parts = list(geom[uid].geoms) if geom[uid].geom_type == 'MultiPolygon' else [geom[uid]]
        res = {v: [] for v in VARIANTS}
        wts = []
        sets = []
        if len(parts) > 1:
            for part in parts:
                m = shapely.contains_xy(part, xy[:, 0], xy[:, 1])
                if m.sum() >= 50:
                    sets.append((xy[m], part.area))
        if not sets:
            sets = [(xy, geom[uid].area)]
        for s, parea in sets:
            n, ordered, hull = counts(s)
            res['doc'].append(L_of(hull / n ** 2 * ordered))
            res['n_nm1'].append(L_of(hull / (n * (n - 1)) * ordered))
            res['unordered'].append(L_of(hull / n ** 2 * ordered / 2))
            res['pairs_over_half'].append(L_of(hull / (n * (n - 1) / 2) * ordered))
            res['census_area'].append(L_of(parea / n ** 2 * ordered))
            wts.append(parea)
        n, ordered, hull = counts(xy)
        row = dict(city=city, unit=uid, n=len(g), area=geom[uid].area, inc=g.inc.iloc[0])
        for v in VARIANTS:
            Lv = L_of(hull / n ** 2 * ordered) if v == 'whole_unit' else np.average(np.vstack(res[v]), axis=0, weights=wts)
            row[f'{v}_L2'], row[f'{v}_L50'], row[f'{v}_auc'] = Lv[0], Lv[-1], np.trapezoid(Lv, R)
        rows.append(row)
D = pd.DataFrame(rows)
lev, tests = [], []
for city, d in D[D.inc.notna()].groupby('city'):
    d = d.copy(); d['q'] = pd.qcut(d.inc, 4, labels=[1, 2, 3, 4]).astype(int)
    for v in VARIANTS:
        s = spearmanr(d[f'{v}_auc'], d.inc)
        kw = kruskal(*[g[f'{v}_auc'] for _, g in d.groupby('q')])
        tests.append(dict(city=city, variant=v, n=len(d), spearman_rho=s.statistic, spearman_p=s.pvalue, kruskal_H=kw.statistic,
                          kruskal_p=kw.pvalue))
        for q, gq in d.groupby('q'):
            l2 = np.average(gq[f'{v}_L2'], weights=gq.area); l50 = np.average(gq[f'{v}_L50'], weights=gq.area)
            lev.append(dict(city=city, variant=v, quartile=q, L2=l2, L50=l50, pub_L2=PUB[(city, q)][0], pub_L50=PUB[(city, q)][1]))
lev = pd.DataFrame(lev)
lev['abs_err'] = (lev.L2 - lev.pub_L2).abs() + (lev.L50 - lev.pub_L50).abs()
lev.to_csv(f'{OUT}/discrepancy_levels.csv', index=False)
pd.DataFrame(tests).to_csv(f'{OUT}/discrepancy_tests.csv', index=False)
pd.set_option('display.width', 200)
print(lev.groupby('variant').abs_err.agg(['mean', 'max']).sort_values('mean').round(2))
print(pd.DataFrame(tests).round(4).to_string(index=False))
