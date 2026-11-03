"""
build_unit_table.py -- census-unit analysis tables for ONE tree dataset, with the manuscript's rules.

Writes into $TREE_MAPPING_BASE_DIR/analysis_output/equity_results/ the three tables the original scripts read:
  all_cities_urban_covariates.csv, all_cities_urban_covariates_corrected.csv, all_cities_park_covariates.csv
Tree columns are recomputed from the dataset's points (never copied from the committed tables):
  unit membership     point attribute block_group_id (centroid-in-unit, first match: the original rule)
  analysis units      units with income (median_household_income on the points) and >= 1 detected tree
  tree_count          number of points;  total_crown_area_m2 = sum of crown_area_m2 (projected polygon areas)
  tree_density        tree_count / block_area_km2, capped at 30,000 (Supplementary Note 3)
  canopy_cover_pct_corrected  100 * total_crown_area_m2 / (block_area_km2 * 1e6)   (summed crown areas, as published)
  mean/median crown area      only for units with >= 10 trees (else NaN)
  income_zscore       (income - mean) / sd over the city's analysis units (pandas sd, ddof = 1)
  income_quartile     equal-frequency quartiles of income within city over the analysis units
  log10 columns       log10(y + 1)
Non-tree covariates (block area, housing units/density, distance to centre, parks, income) are taken from the committed
tables for units that are in them, after checking the dataset's income equals the committed income; for units new to
the analysis they are rebuilt with rules verified to reproduce the committed values exactly:
  block_area_km2      full census-unit geometry (TIGER 2023 BG / StatCan 2021 DA), area_deg2 * 12391 * cos(lat)
  dist_cbd_km         UTM distance from the municipality-clipped unit centroid to the city reference point
                      (reference points recovered from the committed distances, residual 0 m)
  housing_units       Surrey: StatCan 2021 characteristic 4 'Total private dwellings' (reproduces all 573 committed)
  park_pct            municipal park layers (Surrey layers 256 + 245, downloaded 2026-11-01; reproduce 99.5 % of the
                      committed DA values exactly, max drift 0.94 percentage points)
Usage: TREE_MAPPING_DATASET_DIR=... TREE_MAPPING_BASE_DIR=... python build_unit_table.py
"""
import os, sys, json
import numpy as np, pandas as pd, geopandas as gpd, pyogrio
from shapely.validation import make_valid

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'repo_adapted'))
import repro_data  # noqa: E402

ROOT = '/path/to/Urban_data'
NONTREE = os.path.join(HERE, 'repo_adapted', 'data', 'covariates_nontree')
PARK = repro_data.COMMITTED_PARK_CSV
BND = repro_data.COMMITTED_BOUNDARIES_DIR
DERIVED = os.path.join(HERE, 'derived')
CITIES = ['austin', 'bloomington', 'cupertino', 'surrey']
UTM = {'austin': 32614, 'bloomington': 32616, 'cupertino': 32610, 'surrey': 32610}
CBD = {'austin': (621154.36, 3349896.43), 'bloomington': (540282.91, 4335379.26),
       'cupertino': (585746.52, 4131143.89), 'surrey': (511009.71, 5448599.63)}
TIGER = {'austin': '48', 'bloomington': '18', 'cupertino': '06'}
DENSITY_CAP = 30000.0
OUT = repro_data.EQUITY_STAGE_DIR
os.makedirs(OUT, exist_ok=True)

cov0 = pd.read_csv(f'{NONTREE}/all_cities_urban_covariates.csv')
park0 = pd.read_csv(PARK)
NONTREE_COLS = ['GEOID', 'city', 'clean_id', 'median_income', 'housing_units', 'median_year_built', 'block_area_km2',
                'housing_density_km2', 'dist_cbd_km']


def deg_area_km2(geoms):
    return geoms.area * 12391.0 * np.cos(np.radians(geoms.centroid.y))


def new_unit_covariates(city, ids):
    """Non-tree covariates for units absent from the committed tables (rules verified on the committed units)."""
    idc = 'DAUID' if city == 'surrey' else 'GEOID'
    ex = gpd.read_file(f'{BND}/{city}_exact_block_groups.geojson'); ex['id'] = ex[idc].astype('int64')
    ex = ex[ex.id.isin(ids)].set_index('id')
    if city == 'surrey':
        full = ex.geometry  # DAs lie fully inside Surrey; committed block areas reproduce on this geometry (2e-6)
    else:
        t = gpd.read_file(f'{ROOT}/sherlock_venv/notebooks/census_boundaries/tl_2023_{TIGER[city]}_bg/tl_2023_{TIGER[city]}_bg.shp')
        t['id'] = t.GEOID.astype('int64'); full = t.set_index('id').geometry.reindex(ex.index)
    out = pd.DataFrame(index=ex.index)
    out['block_area_km2'] = deg_area_km2(full)
    c = ex.to_crs(UTM[city]).geometry.centroid
    out['dist_cbd_km'] = np.hypot(c.x - CBD[city][0], c.y - CBD[city][1]) / 1000
    if city != 'surrey':
        # US units new to the analysis (v1: Bloomington 181050011032): ACS B25001 is not available locally and the source of
        # the committed US park values is not recorded, so housing and park covariates are left MISSING (not imputed): the
        # unit enters the unadjusted analyses and is dropped (complete case) from the adjusted models. Logged in
        # unit_table_log.csv; flagged on the manuscript checklist.
        for c in ('housing_units', 'housing_density_km2', 'median_year_built', 'unit_clipped_area_m2', 'park_area_m2',
                  'natural_area_m2'):
            out[c] = np.nan
        out['park_source'] = 'MISSING: new US unit, covariate source unavailable (complete-case exclusion from adjusted models)'
        print(f'[units] {city}: new units {sorted(ids)} have no housing/park covariates (adjusted models exclude them)')
        return out
    dw = pd.read_csv(f'{DERIVED}/surrey_statcan_total_private_dwellings.csv').set_index('id').v
    out['housing_units'] = dw.reindex(out.index)
    out['housing_density_km2'] = out.housing_units / out.block_area_km2
    out['median_year_built'] = np.nan
    p = [gpd.read_file(f'{ROOT}/tree_crowns_v1_20261101/provenance/external_downloads/surrey_layer_{L}.geojson').to_crs(32610)
         for L in (256, 245)]
    tot = make_valid(gpd.GeoSeries(list(p[0].geometry) + list(p[1].geometry), crs=32610).union_all())
    nat = make_valid(p[1].geometry.union_all())
    g = ex.to_crs(32610).geometry.apply(lambda x: make_valid(x) if not x.is_valid else x)
    out['unit_clipped_area_m2'] = g.area
    out['park_area_m2'] = [x.intersection(tot).area for x in g]
    out['natural_area_m2'] = [x.intersection(nat).area for x in g]
    out['park_source'] = 'Surrey OpenData layers 256+245 downloaded 2026-11-01 (new unit)'
    return out


cov_rows, park_rows, log = [], [], []
for city in CITIES:
    pts = pyogrio.read_dataframe(repro_data.ensure_city_points(city), read_geometry=False,
                                 columns=['block_group_id', 'median_household_income', 'crown_area_m2', 'crown_area_px']
                                 if 'crown_area_m2' in pyogrio.read_info(repro_data.ensure_city_points(city))['fields']
                                 else ['block_group_id', 'median_household_income', 'crown_area_px', 'latitude'])
    repro_data.apply_crown_area_correction(pts, city)
    pts['median_household_income'] = pd.to_numeric(pts.median_household_income, errors='coerce')
    v = pts.dropna(subset=['block_group_id', 'median_household_income']).copy()
    v['GEOID'] = v.block_group_id.astype('int64')
    agg = v.groupby('GEOID').agg(tree_count=('crown_area_m2', 'size'), total_crown_area_m2=('crown_area_m2', 'sum'),
                                 mean_crown_area_m2=('crown_area_m2', 'mean'), median_crown_area_m2=('crown_area_m2', 'median'),
                                 total_crown_area_px=('crown_area_px', 'sum'), mean_crown_area_px=('crown_area_px', 'mean'),
                                 median_income_pts=('median_household_income', 'first'))
    base = cov0[cov0.city == city][NONTREE_COLS].set_index('GEOID')
    known = agg.index.isin(base.index)
    # committed income must equal the dataset's census attribute for every retained unit
    d = (agg.median_income_pts[known] - base.median_income.reindex(agg.index[known])).abs()
    assert (d < 0.51).all(), f'{city}: income differs from committed table for {int((d >= 0.51).sum())} units'
    t = agg.join(base, how='left')
    pk = park0[park0.city == city].set_index('GEOID')
    t = t.join(pk[['unit_clipped_area_m2', 'park_area_m2', 'park_pct', 'natural_area_m2', 'natural_area_pct']], how='left')
    t['park_source'] = np.where(t.index.isin(pk.index), 'committed all_cities_park_covariates.csv', None)
    new = agg.index[~known]
    if len(new):
        nc = new_unit_covariates(city, set(new))
        for c in nc.columns:
            t.loc[new, c] = nc[c]
        t.loc[new, 'median_income'] = agg.median_income_pts[new]
        t.loc[new, 'clean_id'] = new
        t.loc[new, 'park_pct'] = np.minimum(100, 100 * t.loc[new, 'park_area_m2'] / t.loc[new, 'unit_clipped_area_m2'])
        t.loc[new, 'natural_area_pct'] = np.minimum(100, 100 * t.loc[new, 'natural_area_m2'] / t.loc[new, 'unit_clipped_area_m2'])
    t['city'] = city
    t = t.reset_index()
    t['block_area_m2'] = t.block_area_km2 * 1e6
    t['tree_density_per_km2'] = np.minimum(t.tree_count / t.block_area_km2, DENSITY_CAP)
    t.loc[t.tree_count < 10, ['mean_crown_area_m2', 'median_crown_area_m2', 'mean_crown_area_px']] = np.nan
    t['canopy_cover_pct_corrected'] = 100 * t.total_crown_area_m2 / t.block_area_m2
    t['canopy_cover_pct_uncorrected'] = 100 * t.total_crown_area_px / t.block_area_m2
    t['canopy_cover_pct'] = t.canopy_cover_pct_corrected
    t['tree_area_m2'] = t.total_crown_area_m2
    mu, sd = t.median_income.mean(), t.median_income.std()
    t['income_zscore'] = (t.median_income - mu) / sd
    t['income_quartile'] = pd.qcut(t.median_income, 4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']).astype(str)
    t['merge_id'] = t.GEOID
    t['clean_id'] = t.GEOID
    for a, b in [('tree_density_log10', 'tree_density_per_km2'), ('canopy_cover_log10', 'canopy_cover_pct_corrected'),
                 ('housing_density_log10', 'housing_density_km2'), ('mean_crown_area_log10', 'mean_crown_area_m2')]:
        t[a] = np.log10(t[b] + 1.0)
    cov_rows.append(t)
    log.append(dict(city=city, points=len(pts), points_with_unit_and_income=len(v), analysis_units=len(t),
                    units_in_committed_table=int(known.sum()), units_new=len(new), units_committed_not_in_dataset=
                    int((~base.index.isin(agg.index)).sum()), units_ge10_trees=int((t.tree_count >= 10).sum()),
                    density_capped=int((t.tree_count / t.block_area_km2 > DENSITY_CAP).sum()), income_mean=mu, income_sd=sd))
T = pd.concat(cov_rows, ignore_index=True)
assert T[['block_area_km2', 'dist_cbd_km']].notna().all().all()
missing = T.park_source.fillna('').str.startswith('MISSING')  # new US units without a covariate source (see above)
assert T.loc[~missing, ['housing_density_km2', 'park_pct']].notna().all().all()
uncorr = ['GEOID', 'tree_count', 'tree_area_m2', 'block_area_m2', 'canopy_cover_pct', 'tree_density_per_km2', 'median_income',
          'merge_id', 'income_quartile', 'income_zscore', 'city', 'clean_id', 'housing_units', 'median_year_built',
          'block_area_km2', 'housing_density_km2', 'dist_cbd_km', 'mean_crown_area_m2', 'tree_density_log10',
          'canopy_cover_log10', 'housing_density_log10', 'mean_crown_area_log10']
corr = ['GEOID', 'block_area_m2', 'tree_density_per_km2', 'median_income', 'merge_id', 'income_quartile', 'city', 'clean_id',
        'housing_units', 'median_year_built', 'block_area_km2', 'housing_density_km2', 'dist_cbd_km', 'tree_density_log10',
        'canopy_cover_log10', 'housing_density_log10', 'tree_count', 'total_crown_area_m2', 'mean_crown_area_m2',
        'median_crown_area_m2', 'total_crown_area_px', 'mean_crown_area_px', 'mean_crown_area_log10',
        'canopy_cover_pct_corrected', 'canopy_cover_pct_uncorrected', 'income_zscore']
T[uncorr].to_csv(f'{OUT}/all_cities_urban_covariates.csv', index=False)
T[corr].to_csv(f'{OUT}/all_cities_urban_covariates_corrected.csv', index=False)
P = T[['GEOID', 'clean_id', 'city', 'unit_clipped_area_m2', 'park_area_m2', 'park_pct', 'natural_area_m2', 'natural_area_pct',
       'park_source']].copy()
P['unit_clipped_area_km2'] = P.unit_clipped_area_m2 / 1e6
P['park_area_km2'] = P.park_area_m2 / 1e6
P['natural_area_km2'] = P.natural_area_m2 / 1e6
P['developed_park_pct'] = np.maximum(0, P.park_pct - P.natural_area_pct)
P.to_csv(f'{OUT}/all_cities_park_covariates.csv', index=False)
pd.DataFrame(log).to_csv(f'{OUT}/unit_table_log.csv', index=False)
print(pd.DataFrame(log).to_string(index=False))
