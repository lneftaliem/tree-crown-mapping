"""
run_ripley.py -- runs Christian Igel's Ripley's L analysis (process_city_trees.py, github.com/christian-igel/city-trees,
commit 3c57d4c; included unchanged in this folder) on one tree dataset.

Two substitutions, neither of which changes the method:
  * pointpats.l_test is replaced by l_test_exact, which returns the same statistic without building the full pairwise
    distance matrix (pointpats needs about 40 GB of distances plus a 25 x n(n-1)/2 boolean array for the
    100,000-tree census areas). pointpats (2.3.0-2.6.0) l_test(coords, support, n_simulations=0, linearized=True,
    hull=...).statistic is sqrt(2 * #{i<j: d_ij < r} / n / (n / A_bbox) / pi) - r, with A_bbox the bounding box of
    coords (the hull argument does not enter the observed statistic); l_test_exact computes the same quantity with a
    KD-tree (strict d < r via the next float below r). check_exact.py verifies equality with pointpats.
  * np.random is seeded (0) for the 100,000-tree subsample, which the original code leaves unseeded.

Projection: RIPLEY_PROJ=utm (used for the revised Fig. S6) runs process_city_trees_utm.py, which is the original script
with its two to_crs(epsg=3857) calls replaced by the UTM zone of each city (EPSG 32614 Austin, 32616 Bloomington,
32610 Cupertino and Surrey), as proposed by C. Igel. RIPLEY_PROJ=3857 runs the original script as published
(it reproduces the published Fig. S6 on the original data). RIPLEY_AREA=hull is a sensitivity variant only (convex hull
instead of the bounding box that pointpats uses).

Usage: [RIPLEY_PROJ=utm|3857] [RIPLEY_AREA=bbox|hull] python run_ripley.py <work dir> <city> [<city> ...]
<work dir> must hold census_boundaries/<city>_exact_block_groups.geojson and <city>_ripley_points_with_income.geojson;
writes <work dir>/<city>_Ripley.parquet (the original script's output).
"""
import os, sys, runpy
from collections import namedtuple
import numpy as np
from scipy.spatial import cKDTree
import pointpats

HERE = os.path.dirname(os.path.abspath(__file__))
_Res = namedtuple('LtestResult', ('support', 'statistic', 'pvalue', 'simulations'))
AREA = os.environ.get('RIPLEY_AREA', 'bbox')
UTM = {'austin': 32614, 'bloomington': 32616, 'cupertino': 32610, 'surrey': 32610}


def l_test_exact(coordinates, support=None, n_simulations=0, linearized=False, hull=None, **kw):
    xy = np.asarray(coordinates, float)
    r = np.asarray(support, float)
    n = len(xy)
    t = cKDTree(xy)
    ordered = t.count_neighbors(t, np.nextafter(r, 0)).astype(float) - n   # ordered pairs i != j with d < r
    area = np.ptp(xy[:, 0]) * np.ptp(xy[:, 1]) if AREA == 'bbox' else hull.area
    k = ordered / n / (n / area)                                               # = 2 * unordered / n / intensity
    stat = np.sqrt(k / np.pi)
    return _Res(r, stat - r if linearized else stat, np.full(len(r), np.nan), None)


if __name__ == '__main__':
    work, cities = sys.argv[1], sys.argv[2:]
    pointpats.l_test = l_test_exact
    np.random.seed(0)
    os.chdir(work)
    if os.environ.get('RIPLEY_PROJ', 'utm') == '3857':
        sys.argv = [os.path.join(HERE, 'process_city_trees.py')] + cities
        runpy.run_path(sys.argv[0], run_name='__main__')
    else:
        for c in cities:
            sys.argv = [os.path.join(HERE, 'process_city_trees_utm.py'), c]
            runpy.run_path(sys.argv[0], init_globals={'PROJ_EPSG': UTM}, run_name='__main__')
