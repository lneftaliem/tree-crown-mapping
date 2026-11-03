"""
sensitivity.py -- pre-specified sensitivity analyses that leave the primary specification unchanged.

Primary (as published, build_unit_table.py): summed crown areas / full census-unit area; tree density / full area.
  union_canopy        canopy cover from the UNION of the unit's crown polygons (overlapping crown polygons counted once;
                      crowns assigned to units by the primary point membership, polygons not clipped), / full unit area
  clipped_denominator density and summed-crown canopy / area of the unit clipped to the municipal boundary (UTM area)
For each, the unadjusted Spearman test against income_zscore over the same analysis units as the primary table, next to
the primary statistic.
Also counts the census units whose ACS 2019-2023 B19013 median household income is top-coded ('250,000+'): the
reproduction keeps the published treatment (non-numeric -> missing income -> not an analysis unit) and reports the count.
Writes analysis_output/sensitivity/{canopy_denominator_spearman.csv, unit_sensitivity_values.csv, acs_topcoded_units.csv}.
"""
import os, sys
import numpy as np, pandas as pd, geopandas as gpd, pyogrio, shapely
from scipy.stats import spearmanr
from shapely.validation import make_valid

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'repo_adapted'))
import repro_data  # noqa: E402

OUT = os.path.join(repro_data.BASE_DIR, 'analysis_output', 'sensitivity')
os.makedirs(OUT, exist_ok=True)
UTM = {'austin': 32614, 'bloomington': 32616, 'cupertino': 32610, 'surrey': 32610}
ACS_INC = '/path/to/Urban_data/sherlock_venv/notebooks/income_data/ACSDT5Y2023/ACSDT5Y2023.B19013-Data.csv'
d = pd.read_csv(os.path.join(repro_data.EQUITY_STAGE_DIR, 'all_cities_urban_covariates_corrected.csv'))
uid_col = next(c for c in ('block_group_id', 'GEOID', 'unit_id') if c in d.columns)

vals, tests, top = [], [], []
for city, crs in UTM.items():
    g = d[d.city == city].copy()
    g['uid'] = g[uid_col].astype('int64')
    bfile, bgfile = repro_data.ensure_boundaries(city)
    bg = gpd.read_file(bgfile).to_crs(crs)
    bg['uid'] = bg['DAUID' if city == 'surrey' else 'GEOID'].astype('int64')
    muni = make_valid(gpd.read_file(bfile).to_crs(crs).union_all())
    bg = bg[bg.uid.isin(g.uid)].set_index('uid')
    clipped_km2 = bg.geometry.intersection(muni).area / 1e6
    has_poly = os.path.exists(os.path.join(repro_data.DATASET_DIR, repro_data.POLYGONS_STEM.format(city=city)))
    g = g.set_index('uid')
    if has_poly:  # the original (unsplit) dataset has points only
        poly = pyogrio.read_dataframe(repro_data.ensure_city_polygons(city), columns=['block_group_id']).to_crs(crs)
        poly = poly[poly.block_group_id.notna()]
        poly['uid'] = poly.block_group_id.astype('int64')
        poly = poly[poly.uid.isin(g.index)]
        union_m2 = poly.groupby('uid').geometry.apply(lambda s: shapely.union_all(s.values).area)
        g['union_canopy_pct'] = 100 * union_m2.reindex(g.index).fillna(0) / (g.block_area_km2 * 1e6)
    else:
        g['union_canopy_pct'] = np.nan
    g['clipped_area_km2'] = clipped_km2.reindex(g.index)
    g['density_clipped'] = (g.tree_count / g.clipped_area_km2).clip(upper=30000)
    g['canopy_clipped_pct'] = 100 * g.total_crown_area_m2 / (g.clipped_area_km2 * 1e6)
    g['union_to_summed_ratio'] = g.union_canopy_pct / g.canopy_cover_pct_corrected
    vals.append(g.reset_index()[['city', 'uid', 'tree_count', 'block_area_km2', 'clipped_area_km2', 'tree_density_per_km2',
                                 'density_clipped', 'canopy_cover_pct_corrected', 'union_canopy_pct', 'canopy_clipped_pct',
                                 'union_to_summed_ratio', 'income_zscore']])
    for metric, col in [('canopy cover', 'canopy_cover_pct_corrected'), ('canopy cover', 'union_canopy_pct'),
                        ('canopy cover', 'canopy_clipped_pct'), ('tree density', 'tree_density_per_km2'),
                        ('tree density', 'density_clipped')]:
        s = g.dropna(subset=[col, 'income_zscore'])
        if s.empty:
            continue
        r = spearmanr(s.income_zscore, s[col])
        spec = {'canopy_cover_pct_corrected': 'primary (summed crowns / full unit)', 'tree_density_per_km2': 'primary (full unit)',
                'union_canopy_pct': 'polygon union / full unit', 'canopy_clipped_pct': 'summed crowns / municipality-clipped unit',
                'density_clipped': 'municipality-clipped unit'}[col]
        tests.append(dict(city=city, metric=metric, specification=spec, n=len(s), rho=r.statistic, p=r.pvalue))
    if city != 'surrey':
        acs = pd.read_csv(ACS_INC, skiprows=[1], dtype=str).drop_duplicates('GEO_ID')
        acs['uid'] = acs.GEO_ID.str[-12:].astype('int64')
        allbg = gpd.read_file(bgfile)
        allbg['uid'] = allbg.GEOID.astype('int64')
        a = acs[acs.uid.isin(allbg.uid)]
        pts = pyogrio.read_dataframe(repro_data.ensure_city_points(city), columns=['block_group_id'], read_geometry=False)
        with_trees = set(pts.block_group_id.dropna().astype('int64'))
        tc = a[a.B19013_001E.str.contains(r'\+', na=False)]
        top.append(dict(city=city, units_in_city_file=len(allbg), units_topcoded_250k=len(tc),
                        topcoded_with_detected_trees=int(tc.uid.isin(with_trees).sum()),
                        topcoded_uids=';'.join(map(str, sorted(tc.uid)))))
    else:
        top.append(dict(city=city, units_in_city_file=np.nan, units_topcoded_250k=0, topcoded_with_detected_trees=0,
                        topcoded_uids='(StatCan 2021 DA median income is not top-coded)'))
pd.concat(vals).to_csv(f'{OUT}/unit_sensitivity_values.csv', index=False)
T = pd.DataFrame(tests); T.to_csv(f'{OUT}/canopy_denominator_spearman.csv', index=False)
pd.DataFrame(top).to_csv(f'{OUT}/acs_topcoded_units.csv', index=False)
pd.set_option('display.width', 200)
print(T.round(4).to_string(index=False)); print(pd.DataFrame(top).drop(columns='topcoded_uids').to_string(index=False))
