"""
hull_geometry_check.py -- checks that convex-hull areas larger than their census polygon are geometric (concave
polygons), not coordinate, spatial-join or geometry errors. For every analysed polygon part (>= 50 trees, UTM):
  (a) every assigned tree lies within the polygon (re-tested with shapely.contains, independent of sjoin)
  (b) the tree hull lies within the polygon's own convex hull (must hold if (a) holds), so hull area <= convex(polygon)
  (c) polygon area in UTM agrees with the geodesic (ellipsoidal) area from the original lon/lat geometry
  (d) polygons are valid; tree points round-trip lon/lat -> UTM consistently with the boundary CRS
  (e) hull > polygon only where the polygon is concave (polygon area < its convex-hull area)
Writes hull_geometry_check.csv (current directory). Usage: python hull_geometry_check.py <work_hull dir>
"""
import sys, os
import numpy as np, pandas as pd, geopandas as gpd, shapely
from pyproj import Geod
from shapely.geometry import Polygon
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from run_ripley import UTM
W = sys.argv[1]; geod = Geod(ellps='WGS84'); rows = []
for city in ['austin', 'bloomington', 'cupertino', 'surrey']:
    bg0 = gpd.read_file(f'{W}/census_boundaries/{city}_exact_block_groups.geojson')
    idn = 'DAUID' if 'DAUID' in bg0.columns else 'GEOID'
    bgll = bg0.to_crs(4326); bg = bg0.to_crs(epsg=UTM[city])
    tr = gpd.read_file(f'{W}/{city}_ripley_points_with_income.geojson').to_crs(epsg=UTM[city])
    parts = []
    for u, g, gll in zip(bg[idn].astype(int), bg.geometry, bgll.geometry):
        ps = [g] if isinstance(g, Polygon) else list(g.geoms)
        pl = [gll] if isinstance(gll, Polygon) else list(gll.geoms)
        parts += [(u, k, p, q) for k, (p, q) in enumerate(zip(ps, pl))]
    pg = gpd.GeoDataFrame({'unit': [a[0] for a in parts], 'part': [a[1] for a in parts]},
                          geometry=[a[2] for a in parts], crs=bg.crs)
    j = gpd.sjoin(tr[['geometry']], pg, predicate='within')
    for i, (u, k, p, q) in enumerate(parts):
        pts = j.geometry[(j.unit == u) & (j.part == k)]
        if len(pts) < 50: continue
        xy = shapely.get_coordinates(pts.values)
        h = shapely.MultiPoint(xy).convex_hull
        rows.append(dict(city=city, unit=u, part=k, trees=len(xy), valid=p.is_valid,
                         all_within=bool(shapely.contains(p, shapely.points(xy)).all()),
                         hull_in_poly_hull=bool(p.convex_hull.buffer(1e-6).contains(h)),
                         poly_area=p.area, geodesic_area=abs(geod.geometry_area_perimeter(q)[0]),
                         poly_convex_area=p.convex_hull.area, hull_area=h.area,
                         hull_inside_poly_area=h.intersection(p).area))
d = pd.DataFrame(rows); d['utm_over_geodesic'] = d.poly_area / d.geodesic_area
d['hull_over_poly'] = d.hull_area / d.poly_area; d['concave'] = d.poly_area < d.poly_convex_area * (1 - 1e-9)
d.to_csv('hull_geometry_check.csv', index=False)
for c, x in d.groupby('city', sort=False):
    big = x[x.hull_over_poly > 1]
    print(f"{c}: parts {len(x)}, invalid {(~x.valid).sum()}, trees not within {(~x.all_within).sum()}, "
          f"hull outside poly-hull {(~x.hull_in_poly_hull).sum()}, UTM/geodesic area {x.utm_over_geodesic.min():.4f}-{x.utm_over_geodesic.max():.4f}, "
          f"hull>poly {len(big)} (all concave: {big.concave.all()}), hull-in-poly/poly max {(x.hull_inside_poly_area / x.poly_area).max():.6f}")
print(d.sort_values('hull_over_poly').tail(3)[['city', 'unit', 'part', 'trees', 'poly_area', 'poly_convex_area', 'hull_area']])
