"""
run_ripley.py -- runs Christian Igel's Ripley's L analysis (process_city_trees.py, github.com/christian-igel/city-trees,
commit 3c57d4c; included unchanged in this folder) on one tree dataset, with the convex hull of the trees as the area.

His script is run as is (in the UTM version, process_city_trees_utm.py). It splits each census area into its polygons,
assigns trees to each polygon (sjoin, predicate "within"), skips polygons with fewer than 50 trees or a zero-area hull,
computes hull_geom = MultiPoint(coords).convex_hull for each polygon, calls
l_test(coords, sup, n_simulations=0, linearized=True, hull=hull_geom) for r = 2, 4, ..., 50 m, and averages the
polygons of a MultiPolygon with l += area * statistic; l /= area_auc (area = convex-hull area of the polygon's trees).

pointpats.l_test ignores hull= for the observed statistic and uses the bounding box of the coordinates as the area
(pointpats 2.3.0-2.6.0). As agreed with C. Igel, pointpats.l_test is replaced by l_test_hull, the same statistic with
A = hull.area, the convex hull his code passes in:

    K(r) = A * #{(i, j): i != j, d_ij < r} / n**2      (pointpats normalisation)
    L(r) - r = sqrt(K(r) / pi) - r                      (linearized=True)

No edge correction. Pairs are counted with a KD-tree (strict d < r via the next float below r, as in pointpats), which
avoids the full distance matrix pointpats builds (> 100 GB for the largest census areas). With A = bounding box
(l_test_bbox, RIPLEY_AREA=bbox) the statistic equals pointpats exactly; tests_ripley_hull.py checks both.
np.random is seeded (0) for the 100,000-tree subsample, which the original code leaves unseeded (no polygon in the final
data has more than 100,000 trees).

Projection: RIPLEY_PROJ=utm (default) runs process_city_trees_utm.py, the original script with its two
to_crs(epsg=3857) calls replaced by the UTM zone of each city (EPSG 32614 Austin, 32616 Bloomington, 32610 Cupertino
and Surrey), as proposed by C. Igel. RIPLEY_PROJ=3857 runs the original script as published.
Area: RIPLEY_AREA=hull (default, revised Fig. S6) or bbox (pointpats behaviour; with RIPLEY_PROJ=3857 it reproduces the
published Fig. S6 on the original data).

Usage: [RIPLEY_PROJ=utm|3857] [RIPLEY_AREA=hull|bbox] python run_ripley.py <work dir> <city> [<city> ...]
<work dir> must hold census_boundaries/<city>_exact_block_groups.geojson and <city>_ripley_points_with_income.geojson;
writes <work dir>/<city>_Ripley.parquet (the original script's output).
"""
import os, sys, runpy
from collections import namedtuple
import numpy as np
import shapely
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
LtestResult = namedtuple('LtestResult', ('support', 'statistic', 'pvalue', 'simulations'))
UTM = {'austin': 32614, 'bloomington': 32616, 'cupertino': 32610, 'surrey': 32610}


def ordered_pairs_below(xy, r):
    """#{(i, j): i != j, d_ij < r} for each r (strict inequality; duplicate points count as pairs at d = 0)."""
    t = cKDTree(xy)
    return t.count_neighbors(t, np.nextafter(np.asarray(r, float), 0)).astype(float) - len(xy)


def ripley_l(xy, r, area, linearized=True):
    """Observed L(r) (or L(r) - r) with area A, no edge correction, pointpats normalisation (n**2)."""
    xy, r = np.asarray(xy, float), np.asarray(r, float)
    n = len(xy)
    if n < 2 or not np.isfinite(area) or area <= 0:
        raise ValueError(f'need n >= 2 and area > 0 (n = {n}, area = {area})')
    stat = np.sqrt(area * ordered_pairs_below(xy, r) / n ** 2 / np.pi)
    return stat - r if linearized else stat


def bbox_area(xy):
    xy = np.asarray(xy, float)
    return float(np.ptp(xy[:, 0]) * np.ptp(xy[:, 1]))


def l_test_hull(coordinates, support=None, n_simulations=0, linearized=False, hull=None, **kw):
    """Drop-in for pointpats.l_test as process_city_trees.py calls it, with A = hull.area (the hull it passes)."""
    if n_simulations:
        raise NotImplementedError('only the observed statistic (n_simulations=0) is used in this analysis')
    xy = np.asarray(coordinates, float)
    if hull is None:
        hull = shapely.MultiPoint(xy).convex_hull
    r = np.asarray(support, float)
    return LtestResult(r, ripley_l(xy, r, hull.area, linearized), np.full(len(r), np.nan), None)


def l_test_bbox(coordinates, support=None, n_simulations=0, linearized=False, hull=None, **kw):
    """The same with A = bounding box of the coordinates: what pointpats.l_test returns."""
    xy = np.asarray(coordinates, float)
    r = np.asarray(support, float)
    return LtestResult(r, ripley_l(xy, r, bbox_area(xy), linearized), np.full(len(r), np.nan), None)


if __name__ == '__main__':
    import pointpats
    work, cities = sys.argv[1], sys.argv[2:]
    pointpats.l_test = l_test_bbox if os.environ.get('RIPLEY_AREA', 'hull') == 'bbox' else l_test_hull
    np.random.seed(0)
    os.chdir(work)
    if os.environ.get('RIPLEY_PROJ', 'utm') == '3857':
        sys.argv = [os.path.join(HERE, 'process_city_trees.py')] + cities
        runpy.run_path(sys.argv[0], run_name='__main__')
    else:
        for c in cities:
            sys.argv = [os.path.join(HERE, 'process_city_trees_utm.py'), c]
            runpy.run_path(sys.argv[0], init_globals={'PROJ_EPSG': UTM}, run_name='__main__')
