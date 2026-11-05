"""
tests_ripley_hull.py -- validation of the convex-hull statistic in run_ripley.py. Each test prints PASS/FAIL; the script exits 1 on any failure.

  T1 convex-hull area (shapely, as in Christian's code) vs scipy ConvexHull and analytic areas
  T2 ordered-pair counting vs brute force (pdist), incl. ties at exactly d = r and duplicate points
  T3 normalisation and L transformation on a hand-computed example
  T4 bounding-box version == pointpats.l_test (synthetic patterns and real census polygons)
  T5 convex-hull version == pointpats statistic rescaled to the hull area: L_h + r = (L_bb + r) * sqrt(A_h / A_bb)
  T6 CSR sanity check: L(r) - r close to 0 (slightly negative, no edge correction)
  T7 Christian's process_city() (extracted unchanged from process_city_trees_utm.py) on synthetic census areas:
     MultiPolygon aggregation l = sum(A_hull,k * L_k) / sum(A_hull,k), parts < 50 trees, degenerate hulls, empty
     areas, points on a boundary, subsampling
Usage: python tests_ripley_hull.py [<dataset dir with <city>_ripley_points_with_income.geojson>]
"""
import ast, os, sys, warnings
import numpy as np, geopandas as gpd, shapely
from shapely.geometry import Polygon, MultiPolygon, Point
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from pointpats import l_test

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_ripley import ripley_l, ordered_pairs_below, bbox_area, l_test_hull, l_test_bbox, UTM  # noqa
IGEL = HERE

warnings.filterwarnings('ignore', message='n_simulations must be positive')
SUP = np.linspace(2, 50, 25)
rng = np.random.default_rng(20261105)
FAIL = []


def check(name, ok, detail=''):
    print(('PASS ' if ok else 'FAIL ') + name + (f'  [{detail}]' if detail else ''), flush=True)
    if not ok:
        FAIL.append(name)


# ---------------------------------------------------------------- T1 hull area
sq = np.r_[[[0, 0], [100, 0], [100, 100], [0, 100]], rng.uniform(1, 99, (200, 2))]
check('T1a square hull = 10,000 m2', np.isclose(shapely.MultiPoint(sq).convex_hull.area, 1e4, rtol=0, atol=1e-6))
tri = np.r_[[[0, 0], [60, 0], [0, 80]], rng.uniform(0, 1, (100, 2)) * [[20, 20]]]
check('T1b triangle hull = 2,400 m2', np.isclose(shapely.MultiPoint(tri).convex_hull.area, 2400, atol=1e-6))
for i in range(5):
    p = rng.normal(0, 50, (500, 2)) + rng.uniform(-1e5, 1e5, 2) + [5e5, 3e6]   # UTM-like magnitudes
    a, b = shapely.MultiPoint(p).convex_hull.area, ConvexHull(p).volume
    check(f'T1c random pattern {i}: shapely hull area == scipy ConvexHull', np.isclose(a, b, rtol=1e-12), f'{a:.6f}')
line = np.c_[np.arange(60.), 2 * np.arange(60.)]
check('T1d collinear points: hull area 0', shapely.MultiPoint(line).convex_hull.area == 0)
check('T1e identical points: hull area 0', shapely.MultiPoint(np.zeros((60, 2))).convex_hull.area == 0)
check('T1f hull area <= bounding-box area', all(
    shapely.MultiPoint(p).convex_hull.area <= bbox_area(p) + 1e-9 for p in [sq, tri] + [rng.normal(0, 30, (300, 2))
                                                                                        for _ in range(5)]))

# ---------------------------------------------------------------- T2 pair counting
for name, p in [('uniform', rng.uniform(0, 200, (700, 2))),
                ('clustered', np.repeat(rng.uniform(0, 200, (40, 2)), 15, 0) + rng.normal(0, 3, (600, 2))),
                ('grid with ties at d = r', np.array([[x, y] for x in range(0, 60, 2) for y in range(0, 60, 2)], float)),
                ('duplicates', np.r_[rng.uniform(0, 50, (200, 2)), np.repeat([[10., 10.]], 5, 0)])]:
    d = pdist(p)
    brute = np.array([2 * (d < r).sum() for r in SUP], float)
    check(f'T2 ordered pairs, {name}', np.array_equal(ordered_pairs_below(p, SUP), brute))

# ---------------------------------------------------------------- T3 normalisation, L transformation
u = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
r = np.array([0.5, 1.0, 1.2, 1.5])
# d = 1 (4 sides), sqrt 2 (2 diagonals). Strict d < r: r = 1 -> 0 pairs, 1.2 -> 8 ordered, 1.5 -> 12 ordered.
K = 1.0 * np.array([0, 0, 8, 12]) / 4 ** 2
check('T3a K = A * ordered / n^2, L = sqrt(K / pi)', np.allclose(ripley_l(u, r, 1.0, linearized=False), np.sqrt(K / np.pi)))
check('T3b linearized: L - r', np.allclose(ripley_l(u, r, 1.0), np.sqrt(K / np.pi) - r))
check('T3c L scales with sqrt(A)', np.allclose(ripley_l(u, r, 4.0, False), 2 * ripley_l(u, r, 1.0, False)))
for bad in [0.0, -1.0, np.nan]:
    try:
        ripley_l(u, r, bad); ok = False
    except ValueError:
        ok = True
    check(f'T3d area {bad} raises', ok)

# ---------------------------------------------------------------- T4 / T5 pointpats equality (synthetic)
synth = {'CSR': rng.uniform(0, 300, (800, 2)) + [6e5, 3.3e6],
         'clustered': np.repeat(rng.uniform(0, 300, (50, 2)), 16, 0) + rng.normal(0, 4, (800, 2)) + [6e5, 3.3e6],
         'regular': np.array([[x, y] for x in range(0, 120, 4) for y in range(0, 120, 4)], float) + rng.normal(0, .3, (900, 2)),
         'L-shaped': np.r_[rng.uniform(0, [300, 60], (500, 2)), rng.uniform(0, [60, 300], (500, 2))],
         'n = 50': rng.uniform(0, 80, (50, 2))}


def t45(name, xy):
    hull = shapely.MultiPoint(xy).convex_hull
    pp = l_test(xy, SUP, n_simulations=0, linearized=True, hull=hull).statistic
    pp_none = l_test(xy, SUP, n_simulations=0, linearized=True).statistic
    bb = l_test_bbox(xy, SUP, n_simulations=0, linearized=True, hull=hull).statistic
    hh = l_test_hull(xy, SUP, n_simulations=0, linearized=True, hull=hull).statistic
    ratio = np.sqrt(hull.area / bbox_area(xy))
    check(f'T4 {name}: pointpats ignores hull= (hull vs none identical)', np.array_equal(pp, pp_none))
    check(f'T4 {name}: bbox version == pointpats', np.allclose(bb, pp, rtol=0, atol=1e-9), f'max {np.max(np.abs(bb - pp)):.1e}')
    check(f'T5 {name}: hull version == (pointpats + r) * sqrt(A_h/A_bb) - r',
          np.allclose(hh, (pp + SUP) * ratio - SUP, rtol=0, atol=1e-9), f'A_h/A_bb = {ratio ** 2:.3f}')


for k, v in synth.items():
    t45(k, v)

# ---------------------------------------------------------------- T4 / T5 real census polygons
if len(sys.argv) > 1:
    import pyogrio, pyproj
    for city in ['austin', 'bloomington', 'cupertino', 'surrey']:
        p = pyogrio.read_dataframe(f'{sys.argv[1]}/{city}_ripley_points_with_income.geojson',
                                   columns=['block_group_id', 'longitude', 'latitude'], read_geometry=False)
        x, y = pyproj.Transformer.from_crs(4326, UTM[city], always_xy=True).transform(p.longitude.values, p.latitude.values)
        n = p.groupby('block_group_id').size()
        for unit in rng.choice(n[(n >= 50) & (n <= 6000)].index, 3, replace=False):
            m = (p.block_group_id == unit).values
            t45(f'{city} unit {unit} (n = {m.sum()})', np.c_[x[m], y[m]])

# ---------------------------------------------------------------- T6 CSR sanity
# Without edge correction, E[K_hat] for n uniform points in an s x s square, with area A_h, is
#   (n - 1) / n * A_h / s**4 * integral_{|h| < r} (s - |h_x|)(s - |h_y|) dh = (n-1)/n * A_h/s**4 * (pi r^2 s^2 - 8 r^3 s / 3 + r^4 / 2)
s, nb = 1000.0, 20000
big = rng.uniform(0, s, (nb, 2))
lr = l_test_hull(big, SUP, linearized=True).statistic
ah = shapely.MultiPoint(big).convex_hull.area
ek = (nb - 1) / nb * ah / s ** 4 * (np.pi * SUP ** 2 * s ** 2 - 8 * SUP ** 3 * s / 3 + SUP ** 4 / 2)
el = np.sqrt(ek / np.pi) - SUP
check('T6 CSR (20,000 points, 1 km2): L - r matches the analytic expectation without edge correction (|diff| < 0.1 m)',
      np.max(np.abs(lr - el)) < 0.1,
      f'observed {lr[0]:.2f} .. {lr[-1]:.2f} m, expected {el[0]:.2f} .. {el[-1]:.2f} m, max |diff| {np.max(np.abs(lr - el)):.3f}')

# ---------------------------------------------------------------- T7 Christian's process_city() on synthetic areas
src = open(os.path.join(IGEL, 'process_city_trees_utm.py')).read()
tree = ast.parse(src)
keep = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom)) or
        (isinstance(n, ast.FunctionDef) and n.name == 'process_city')]
ns = {'PROJ_EPSG': {'synthetic': 32614}, 'city_name': 'synthetic'}
import pointpats  # noqa: E402
pointpats.l_test = l_test_hull
exec(compile(ast.Module(keep, []), 'process_city_trees_utm.py', 'exec'), ns)   # his function, unchanged
import logging; logging.getLogger().setLevel(logging.WARNING)

E0, N0 = 620000.0, 3350000.0   # metres, UTM 14N (Austin)
def box(x0, y0, w, h):
    return Polygon([(E0 + x0, N0 + y0), (E0 + x0 + w, N0 + y0), (E0 + x0 + w, N0 + y0 + h), (E0 + x0, N0 + y0 + h)])
def pts(x0, y0, w, h, n, clustered=False):
    if clustered:
        c = rng.uniform([x0 + 10, y0 + 10], [x0 + w - 10, y0 + h - 10], (n // 10, 2))
        return np.clip(np.repeat(c, 10, 0) + rng.normal(0, 2, (n // 10 * 10, 2)), [x0 + .5, y0 + .5], [x0 + w - .5, y0 + h - .5])
    return rng.uniform([x0 + .5, y0 + .5], [x0 + w - .5, y0 + h - .5], (n, 2))

A = [box(0, 0, 400, 400), box(1000, 0, 200, 600), box(2000, 0, 300, 300), box(3000, 0, 300, 300)]
parts = {  # unit 1: MultiPolygon, two analysed parts + one part with 49 trees + one part with collinear trees
    (1, 0): pts(0, 0, 400, 400, 1500), (1, 1): pts(1000, 0, 200, 600, 700, True),
    (1, 2): pts(2000, 0, 300, 300, 49), (1, 3): np.c_[np.linspace(3010, 3290, 80), np.full(80, 150.)]}
units = [(1, MultiPolygon(A), 50000)]
units.append((2, box(5000, 0, 300, 300), 70000)); parts[(2, 0)] = pts(5000, 0, 300, 300, 50)          # exactly 50
units.append((3, box(6000, 0, 300, 300), 90000)); parts[(3, 0)] = pts(6000, 0, 300, 300, 30)          # too few
units.append((4, box(7000, 0, 300, 300), 60000))                                                       # no trees
units.append((5, MultiPolygon([box(8000, 0, 300, 300), box(8500, 0, 300, 300)]), 40000))              # both small
parts[(5, 0)] = pts(8000, 0, 300, 300, 40); parts[(5, 1)] = pts(8500, 0, 300, 300, 45)
units.append((6, box(9000, 0, 300, 300), 80000))                                                      # boundary pts
parts[(6, 0)] = np.r_[pts(9000, 0, 300, 300, 60), [[9000, 100], [9300, 200]]]                         # on the edge
rows = []
for (u, k), xy in parts.items():
    inc = {uu: ii for uu, _, ii in units}[u]
    rows += [dict(block_group_id=u, median_household_income=inc, geometry=Point(E0 + a, N0 + b)) for a, b in xy]
trees = gpd.GeoDataFrame(rows, crs=32614)
city = gpd.GeoDataFrame({'GEOID': [u for u, _, _ in units]}, geometry=[g for _, g, _ in units], crs=32614)
out, _ = ns['process_city'](city, trees, 'GEOID', sup=SUP, min_trees=50)
out = out.set_index('GEOID')


def expected(unit, keys, sub=None):
    num, den, used = np.zeros(len(SUP)), 0.0, 0
    for k in keys:
        xy = np.c_[E0 + parts[(unit, k)][:, 0], N0 + parts[(unit, k)][:, 1]]
        a = shapely.MultiPoint(xy).convex_hull.area
        num += a * ripley_l(xy, SUP, a); den += a; used += len(xy)
    return num / den, den, used


l1, a1, n1 = expected(1, [0, 1])
check('T7a MultiPolygon: l = (A1 L1 + A2 L2) / (A1 + A2), with A = convex-hull areas',
      np.allclose(out.loc[1, 'L'], l1, atol=1e-10))
check('T7b MultiPolygon: processed area = sum of hull areas of analysed parts (not polygon areas)',
      np.isclose(out.loc[1, 'processed area'], a1) and not np.isclose(a1, A[0].area + A[1].area),
      f'{a1:,.0f} m2 hull vs {A[0].area + A[1].area:,.0f} m2 polygons')
check('T7c MultiPolygon: trees counts all parts (2,329), processed only analysed parts (2,200)',
      out.loc[1, 'trees'] == 1500 + 700 + 49 + 80 and out.loc[1, 'processed'] == 2200)
from sklearn import metrics  # noqa
check('T7d AUC = trapezoid area of L(r) - r over 2-50 m', np.isclose(out.loc[1, 'auc'], metrics.auc(SUP, l1)))
l2, a2, _ = expected(2, [0])
check('T7e exactly 50 trees: analysed', np.allclose(out.loc[2, 'L'], l2) and out.loc[2, 'processed'] == 50)
for u, why in [(3, '30 trees'), (4, 'no trees'), (5, 'all parts < 50 trees')]:
    check(f'T7f unit {u} ({why}): L and AUC NaN, processed 0 -> excluded by the notebook filter',
          np.all(np.isnan(out.loc[u, 'L'])) and np.isnan(out.loc[u, 'auc']) and out.loc[u, 'processed'] == 0)
check('T7g trees exactly on the polygon boundary are not "within" and are dropped',
      out.loc[6, 'trees'] == 60, f"{out.loc[6, 'trees']} of 62")
# subsampling (max_trees): Christian keeps the hull (and its area) of all trees and uses n = max_trees
np.random.seed(3)
out_s, _ = ns['process_city'](city.iloc[[0]], trees, 'GEOID', sup=SUP, max_trees=1000, min_trees=50)
np.random.seed(3)
xy0 = np.c_[E0 + parts[(1, 0)][:, 0], N0 + parts[(1, 0)][:, 1]]
idx = np.random.choice(1500, size=1000, replace=False)
a0 = shapely.MultiPoint(xy0).convex_hull.area
xy1 = np.c_[E0 + parts[(1, 1)][:, 0], N0 + parts[(1, 1)][:, 1]]
a1b = shapely.MultiPoint(xy1).convex_hull.area
exp_s = (a0 * ripley_l(xy0[idx], SUP, a0) + a1b * ripley_l(xy1, SUP, a1b)) / (a0 + a1b)
check('T7h subsample (> max_trees): n = max_trees, A = hull of all trees in the part',
      np.allclose(out_s.iloc[0]['L'], exp_s) and out_s.iloc[0]['processed'] == 1700)

print(f'\n{len(FAIL)} failure(s)' + (': ' + ', '.join(FAIL) if FAIL else ''))
sys.exit(1 if FAIL else 0)
