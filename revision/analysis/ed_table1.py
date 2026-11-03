"""
ed_table1.py -- Extended Data Table 1: City | Trees identified | Municipal land area (km2) | Census units with detections |
Median density | Maximum density (trees km-2).

The original table-building code (city_tree_analysis.generate_nature_summary_table) took its statistics from an earlier
run whose inputs are not preserved; on the committed original data no documented rule reproduces the published medians
exactly (closest rule, used here, = the Figure 1 rule: Austin 1,085 vs 1,089; Bloomington 1,023 vs 1,014; Cupertino 1,735
vs 1,736; Surrey 3,205 vs 2,903). Rule (identical to generate_density_map_from_geojson.py, Figure 1):
  trees            all final crowns of the city (points GeoJSON)
  units            census units (exact_block_groups) with >= 1 tree, trees joined by point-in-polygon ('intersects',
                   first match), including units without income
  density          trees / municipality-clipped unit area, area_deg2 * 12391 * cos(lat)
  land area        municipal boundary area in the city UTM zone (861.9 / 60.6 / 29.4 / 323.8; published 861.9 / 60.7 / 29.4 /
                   323.9 -- within 0.1 km2; land area does not depend on the tree data)
Writes analysis_output/extended_data/ED_Table1.{csv,md} and per-unit densities ED_Table1_unit_density.csv.
"""
import os, sys
from decimal import Decimal, ROUND_HALF_UP
import numpy as np, pandas as pd, geopandas as gpd, pyogrio
from shapely.validation import make_valid

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'repo_adapted'))
import repro_data  # noqa: E402

OUT = os.path.join(repro_data.BASE_DIR, 'analysis_output', 'extended_data')
os.makedirs(OUT, exist_ok=True)
CITIES = ['austin', 'bloomington', 'cupertino', 'surrey']
LABEL = {'austin': 'Austin, TX', 'bloomington': 'Bloomington, IN', 'cupertino': 'Cupertino, CA', 'surrey': 'Surrey, BC'}
UTM = {'austin': 32614, 'bloomington': 32616, 'cupertino': 32610, 'surrey': 32610}


def half_up(x, nd=1):
    return float(Decimal(str(x)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


rows, units = [], []
for city in CITIES:
    bfile, bgfile = repro_data.ensure_boundaries(city)
    bnd = make_valid(gpd.read_file(bfile).to_crs(4326).union_all())
    bg = gpd.read_file(bgfile).to_crs(4326)
    bg['geometry'] = bg.geometry.apply(lambda g: make_valid(g) if not g.is_valid else g)
    bg['uid'] = bg['DAUID' if city == 'surrey' else 'GEOID'].astype('int64')
    p = pyogrio.read_dataframe(repro_data.ensure_city_points(city), columns=[])
    j = gpd.sjoin(p.to_crs(4326), bg[['uid', 'geometry']], how='inner', predicate='intersects')
    j = j[~j.index.duplicated(keep='first')]
    n = j.groupby('uid').size()
    clip = bg.set_index('uid').geometry.intersection(bnd)
    area = clip.area * 12391.0 * np.cos(np.radians(clip.centroid.y))
    d = pd.DataFrame({'trees': n}).join(area.rename('clipped_area_km2'))
    d['density'] = d.trees / d.clipped_area_km2
    d['city'] = city
    units.append(d.reset_index())
    land = gpd.GeoSeries([bnd], crs=4326).to_crs(UTM[city]).area.iloc[0] / 1e6
    rows.append({'City': LABEL[city], 'Trees identified': len(p), 'Municipal land area (km2)': half_up(land),
                 'Census units with detections': len(d), 'Median density (trees km-2)': d.density.median(),
                 'Maximum density (trees km-2)': d.density.max(), 'trees_outside_all_units': len(p) - int(n.sum())})
T = pd.DataFrame(rows)
T.to_csv(f'{OUT}/ED_Table1.csv', index=False)
pd.concat(units).to_csv(f'{OUT}/ED_Table1_unit_density.csv', index=False)
md = ['| City | Trees identified | Municipal land area (km²) | Census units with detections | Median density (trees km⁻²) | Maximum density (trees km⁻²) |',
      '|---|---:|---:|---:|---:|---:|']
for r in rows:
    md.append(f"| {r['City']} | {r['Trees identified']:,} | {r['Municipal land area (km2)']:.1f} | {r['Census units with detections']} | "
              f"{r['Median density (trees km-2)']:,.0f} | {r['Maximum density (trees km-2)']:,.0f} |")
open(f'{OUT}/ED_Table1.md', 'w').write('\n'.join(md) + '\n')
print('\n'.join(md))
