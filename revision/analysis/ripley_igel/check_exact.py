"""
check_exact.py -- verifies that run_ripley.l_test_bbox (bounding-box area) returns exactly the statistic of pointpats.l_test as Christian
Igel's process_city_trees.py calls it (n_simulations=0, linearized=True, hull=convex hull, r = 2, 4, ..., 50), on
3 random census areas per city with 50-6,000 trees (pointpats builds the full distance matrix, so larger areas are
skipped here). Prints the maximum absolute difference per area; expected 0.
Usage: python check_exact.py <dataset dir with <city>_ripley_points_with_income.geojson>
"""
import os, sys
import numpy as np, pyogrio, pyproj
from pointpats import l_test
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_ripley import l_test_bbox, UTM  # noqa

D = sys.argv[1]
sup = np.linspace(2, 50, 25)
rng = np.random.default_rng(1)
worst = 0
for city in ['austin', 'bloomington', 'cupertino', 'surrey']:
    p = pyogrio.read_dataframe(f'{D}/{city}_ripley_points_with_income.geojson',
                               columns=['block_group_id', 'longitude', 'latitude'], read_geometry=False)
    x, y = pyproj.Transformer.from_crs(4326, UTM[city], always_xy=True).transform(p.longitude.values, p.latitude.values)
    n = p.groupby('block_group_id').size()
    for u in rng.choice(n[(n >= 50) & (n <= 6000)].index, 3, replace=False):
        m = (p.block_group_id == u).values
        xy = np.c_[x[m], y[m]]
        a = l_test(xy, sup, n_simulations=0, linearized=True, hull='convex').statistic
        b = l_test_bbox(xy, sup, n_simulations=0, linearized=True).statistic
        d = float(np.max(np.abs(a - b))); worst = max(worst, d)
        print(city, u, m.sum(), f'max |diff| = {d:.2e}', flush=True)
print('WORST', worst)
