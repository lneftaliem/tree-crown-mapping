"""
11_validate_outputs.py -- independent checks of the generated point/polygon GeoJSONs.

Checks per city: counts and tree_id equality points vs polygons, uniqueness, CRS, polygon validity,
crown_area_m2 recomputed from the polygon in UTM, crown_area_px integer and = pixel count of the
label in the split raster (sampled), centroid-in-municipality, point = polygon centroid,
attributes equal to the original block-group values, missingness, distributions vs original,
>=3 m2 retention by income quintile (split and original, same tiles), unresolved-canopy flags.
Writes diagnostics/final_validation_*.csv and final_validation.json.
Usage: python 11_validate_outputs.py <dir>
"""
import sys, json
import numpy as np, pandas as pd, geopandas as gpd, pyogrio, shapely, rasterio
from shapely.validation import make_valid

import os
ROOT = os.environ.get('URBAN_DATA', '/path/to/Urban_data')
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402
AUD = paths.DIAG
CENSUS = paths.CENSUS
EPSG = {'austin': 32614, 'bloomington': 32616, 'cupertino': 32610, 'surrey': 32610}
ATTR = ['median_household_income', 'has_income', 'total_pop', 'pct_white', 'pct_black', 'pct_asian',
        'pct_hispanic_latino', 'pct_indigenous', 'pct_other_multiracial']
ORIG16 = ['city', 'block_group_id', 'median_household_income', 'has_income', 'longitude', 'latitude', 'x_meters',
          'y_meters', 'crown_area_px', 'total_pop', 'pct_white', 'pct_black', 'pct_asian', 'pct_hispanic_latino',
          'pct_indigenous', 'pct_other_multiracial']
out_dir = sys.argv[1]
G = pd.read_csv(f'{AUD}/georef_table.csv').set_index('filename')
art = pd.read_csv(f'{AUD}/artifact_components.csv')[['tile', 'parent_cc8']].assign(art=True)
res, miss_rows, dist_rows, inc_rows = {}, [], [], []
for city in ['austin', 'bloomington', 'cupertino', 'surrey']:
    r = {}
    p = gpd.read_file(f'{out_dir}/{city}_ripley_points_with_income.geojson')
    q = gpd.read_file(f'{out_dir}/{city}_ripley_polygons_with_income.geojson')
    r['n_points'], r['n_polygons'] = len(p), len(q)
    r['ids_identical_same_order'] = bool((p.tree_id.values == q.tree_id.values).all())
    r['ids_unique'] = bool(p.tree_id.is_unique)
    r['crs'] = str(p.crs), str(q.crs)
    r['all_16_original_fields_present'] = all(c in p.columns for c in ORIG16)
    r['attribute_tables_identical'] = bool(p.drop(columns='geometry').equals(q.drop(columns='geometry')))
    r['polygons_valid'] = bool(q.is_valid.all()); r['polygon_types'] = q.geom_type.value_counts().to_dict()
    qu = q.to_crs(EPSG[city])
    rel = np.abs(qu.area.values - q.crown_area_m2.values) / q.crown_area_m2.values
    r['max_rel_diff_area_attr_vs_polygon'] = float(rel.max())
    r['min_crown_area_m2'] = float(q.crown_area_m2.min())
    r['crown_area_px_integer'] = bool((q.crown_area_px % 1 == 0).all())
    c = qu.centroid
    r['max_point_vs_polygon_centroid_m'] = float(np.hypot(c.x.values - p.x_meters.values, c.y.values - p.y_meters.values).max())
    pu = p.to_crs(EPSG[city])
    r['max_point_geom_vs_xy_m'] = float(np.hypot(pu.geometry.x - p.x_meters, pu.geometry.y - p.y_meters).max())
    bnd = make_valid(gpd.read_file(f'{CENSUS}/{city}_exact_boundary.geojson').to_crs(4326).union_all())
    r['all_centroids_in_municipality'] = bool(shapely.contains_xy(bnd, p.longitude.values, p.latitude.values).all())
    r['polygons_extending_beyond_boundary'] = int((~q.within(bnd)).sum())
    # pixel counts vs raster (sample of up to 300 crowns, all 4 cities)
    smp = q.sample(min(300, len(q)), random_state=0)
    ok = 0
    for tile, g in smp.groupby('source_tile'):
        with rasterio.open(paths.split_path(tile)) as src:
            cnt = np.bincount(src.read(1).ravel())
        ok += int((cnt[g.instance_label.values] == g.crown_area_px.values).sum())
    r['pixel_count_matches_raster_sample'] = f'{ok}/{len(smp)}'
    # artifacts excluded
    split = pd.read_parquet(f'{AUD}/crowns_city/{city}_split.parquet', columns=['tile', 'label', 'parent_cc8', 'keep_final',
                                                                               'keep_unfiltered', 'inside', 'area_m2', 'npx'])
    split = split.merge(art, on=['tile', 'parent_cc8'], how='left'); split['art'] = split.art.eq(True)
    key = set(zip(p.source_tile, p.instance_label))
    r['artifact_instances_in_output'] = int(sum((t, l) in key for t, l in zip(split.tile[split.art], split.label[split.art])))
    # attributes == original block-group values
    lk = pd.read_csv(f'{AUD}/bg_lookup_{city}.csv').set_index('block_group_id')
    # block groups absent from the original release have no original values to compare (attributes empty,
    # has_income 'no' by the original convention); counted separately so they cannot mask a real mismatch
    known = p.block_group_id.isin(lk.index)
    r['crowns_in_bgs_without_original_values'] = int((~known).sum())
    r['bgs_without_original_values'] = int(p.block_group_id[~known].nunique())
    r['attrs_empty_where_no_original_values'] = bool(p.loc[~known, [a for a in ATTR if a != 'has_income']].isna().all().all()
                                                     and p.has_income[~known].eq('no').all())
    m = p[known].join(lk[ATTR], on='block_group_id', rsuffix='_orig')
    mism = 0
    for a in ATTR:
        x, y = m[a], m[a + '_orig']
        both_na = x.isna() & y.isna()
        eq = (x == y) | both_na
        if x.dtype.kind == 'f':
            eq = eq | np.isclose(x.astype(float), y.astype(float), rtol=0, atol=1e-12)
        mism += int((~eq).sum())
    r['attribute_mismatches_vs_original_bg_values'] = mism
    r['n_block_groups'] = int(p.block_group_id.nunique())
    r['unresolved_canopy_gt250m2'] = int(p.unresolved_canopy.sum())
    r['unresolved_canopy_area_share'] = float(q.crown_area_m2[q.unresolved_canopy].sum() / q.crown_area_m2.sum())
    res[city] = r
    # missingness: new vs original
    orig = pyogrio.read_dataframe(f'{ROOT}/ripley_data_0429206/all_trees/{city}_ripley_points_with_income.geojson',
                                  read_geometry=False, columns=ORIG16)
    for a in ['block_group_id'] + ATTR:
        mn = p[a].isna().mean() if a != 'has_income' else (p[a] == 'no').mean()
        mo = orig[a].isna().mean() if a != 'has_income' else (orig[a] == 'no').mean()
        miss_rows.append({'city': city, 'attribute': a, 'missing_share_split': mn, 'missing_share_original': mo})
    # distributions (tree-weighted) vs original
    for a in ['median_household_income', 'total_pop', 'pct_white', 'pct_black', 'pct_asian', 'pct_hispanic_latino']:
        for lab, s in [('split_final', p[a]), ('original_published', orig[a])]:
            s = pd.to_numeric(s, errors='coerce')
            dist_rows.append({'city': city, 'attribute': a, 'dataset': lab, 'n': int(s.notna().sum()), 'mean': s.mean(),
                              'q25': s.quantile(.25), 'median': s.median(), 'q75': s.quantile(.75)})
    for lab, s in [('split_final', q.crown_area_m2), ('split_final_excl_unresolved', q.crown_area_m2[~q.unresolved_canopy])]:
        dist_rows.append({'city': city, 'attribute': 'crown_area_m2', 'dataset': lab, 'n': len(s), 'mean': s.mean(),
                          'q25': s.quantile(.25), 'median': s.median(), 'q75': s.quantile(.75), 'q95': s.quantile(.95),
                          'q99': s.quantile(.99), 'max': s.max(), 'total_m2': s.sum()})
    # >= 3 m2 sensitivity by income quintile (same tiles; unfiltered = manuscript dedup, centroid-in-city,
    # artifacts excluded); quintiles of block-group median income within city
    q5 = lk.dropna(subset=['median_household_income']).copy()
    q5['inc_q'] = pd.qcut(q5.median_household_income, 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5']).astype(str)
    qmap = q5.inc_q.to_dict()
    for kind in ('split', 'orig'):
        d = pd.read_parquet(f'{AUD}/crowns_city/{city}_{kind}.parquet', columns=['tile', 'parent_cc8', 'keep_unfiltered',
                                                                                 'area_m2', 'block_group_id'] if kind == 'split'
                            else ['keep_unfiltered', 'area_m2', 'block_group_id'])
        if kind == 'split':
            d = d.merge(art, on=['tile', 'parent_cc8'], how='left'); d = d[~d.art.eq(True)]
        d = d[d.keep_unfiltered]
        d['inc_q'] = d.block_group_id.map(lambda v: qmap.get(int(v), 'income_missing') if pd.notna(v) else 'no_bg')
        for qq, g in d.groupby('inc_q'):
            inc_rows.append({'city': city, 'dataset': kind, 'inc_q': qq, 'n_unfiltered': len(g),
                             'n_ge3m2': int((g.area_m2 >= 3).sum()), 'retained_share': (g.area_m2 >= 3).mean()})
    print(city, r, flush=True)
json.dump(res, open(f'{AUD}/final_validation.json', 'w'), indent=1, default=str)
pd.DataFrame(miss_rows).to_csv(f'{AUD}/final_validation_missingness.csv', index=False)
pd.DataFrame(dist_rows).to_csv(f'{AUD}/final_validation_distributions.csv', index=False)
pd.DataFrame(inc_rows).to_csv(f'{AUD}/final_validation_income_quintile_filter.csv', index=False)
