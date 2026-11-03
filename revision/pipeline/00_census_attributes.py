"""
00_census_attributes.py -- census-unit attribute lookups (bg_lookup_<city>.csv) for the v1 dataset.

The original release attached income and demographics per census unit from authoritative sources:
  US       ACS 2019-2023 5-year B19013 (median household income) and B03002 (race/ethnicity), block groups;
  Surrey   Statistics Canada 2021 Census Profile 98-401-X2021006 (BC), dissemination areas:
           characteristic 243 'Median total income of household in 2020 ($)' (CAD, as stored in the release) and
           characteristics 1403/1683-1697 (visible minority / Indigenous) via sherlock_venv/add_racial_demographics_to_trees.py.
The original lookups (split_audit_20261031/diagnostics_full/bg_lookup_<city>.csv, built from the published
per-tree files) are kept for every unit they contain. Units with crowns in v1 that the original release did not
contain (14 Surrey DAs in the newly covered 186_028 area) are filled from the SAME source with the SAME code, after
verifying that this code reproduces every original Surrey value exactly. Nothing is taken from the incompatible
ripley_data_0429206/all_trees/archive/surrey_blockgroups_with_income.geojson.
Every unit of <city>_exact_block_groups.geojson is covered, so any unit that gains crowns has attributes. US units absent
from the release are filled from B19013 (read as city_tree_analysis.load_us_income_data) and B03002 (demographics code),
after verifying that the same code reproduces every original US value.
Usage: DIAG_DIR=... python 00_census_attributes.py
"""
import os, sys, importlib.util
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths  # noqa: E402
ROOT = paths.ROOT
ORIG = f'{HERE}/inputs/original_bg_lookup'  # census attributes of the original release
OUT = os.environ['DIAG_DIR']
ACS_INC = f'{ROOT}/sherlock_venv/notebooks/income_data/ACSDT5Y2023/ACSDT5Y2023.B19013-Data.csv'
BG = paths.CENSUS
STATCAN = f'{ROOT}/sherlock_venv/notebooks/income_data/98-401-X2021006_BC_CB_eng_CSV/98-401-X2021006_English_CSV_data_BritishColumbia.csv'
ATTR = ['median_household_income', 'has_income', 'total_pop', 'pct_white', 'pct_black', 'pct_asian', 'pct_hispanic_latino',
        'pct_indigenous', 'pct_other_multiracial']
CITIES = ('austin', 'bloomington', 'cupertino', 'surrey')

spec = importlib.util.spec_from_file_location('demo', f'{HERE}/add_racial_demographics_to_trees.py')
demo = importlib.util.module_from_spec(spec); spec.loader.exec_module(demo)  # functions only (main() is __main__-guarded)
demo.CA_CENSUS_PATH = STATCAN
import geopandas as gpd


def us_income():
    df = pd.read_csv(ACS_INC, skiprows=1, dtype=str)
    df = df.iloc[:, :4]; df.columns = ['GEOID_full', 'NAME', 'median_income', 'income_moe']
    df = df.drop_duplicates('GEOID_full')  # the national file repeats every California row with identical values
    df['block_group_id'] = df.GEOID_full.str.replace('1500000US', '', regex=False).astype('int64')
    return pd.to_numeric(df.median_income, errors='coerce').set_axis(df.block_group_id)


def check(chk, inc, dem):
    """assert the authoritative source + code reproduce every original attribute value"""
    v = chk.median_household_income.dropna()
    assert np.allclose(v, inc.reindex(v.index)), 'income mismatch'
    for a in ATTR[2:]:
        x, y = chk[a], dem[a].reindex(chk.index)
        ok = (x.isna() & y.isna()) | np.isclose(x.astype(float), y.astype(float), rtol=0, atol=1e-12)
        assert ok.all(), f'{a} mismatch for {int((~ok).sum())} units'

os.makedirs(OUT, exist_ok=True)
prov = []
for city in CITIES:
    lk = pd.read_csv(f'{ORIG}/bg_lookup_{city}.csv')
    lk['attribute_source'] = 'original release (verbatim)'
    idcol = 'DAUID' if city == 'surrey' else 'GEOID'
    ids = set(gpd.read_file(f'{BG}/{city}_exact_block_groups.geojson', ignore_geometry=True)[idcol].astype('int64'))
    new = sorted(ids - set(lk.block_group_id.astype('int64')))
    if city == 'surrey':
        inc = []
        for c in pd.read_csv(STATCAN, usecols=['ALT_GEO_CODE', 'GEO_LEVEL', 'CHARACTERISTIC_ID', 'C1_COUNT_TOTAL'], chunksize=2_000_000,
                             dtype={'ALT_GEO_CODE': str}, encoding='latin-1'):
            inc.append(c[(c.GEO_LEVEL == 'Dissemination area') & (c.CHARACTERISTIC_ID == 243) & c.ALT_GEO_CODE.str.startswith('5915')])
        inc = pd.concat(inc).assign(block_group_id=lambda d: d.ALT_GEO_CODE.astype('int64'),
                                    inc=lambda d: pd.to_numeric(d.C1_COUNT_TOTAL, errors='coerce')).set_index('block_group_id').inc
        dem = pd.DataFrame.from_dict(demo.load_ca_demographics(), orient='index')
        src = 'StatCan 98-401-X2021006 char 243 + 1403/1683-1697 (recovered; same source and code as the release)'
    else:
        inc = us_income() if new else None
        dem = pd.DataFrame.from_dict(demo.load_us_demographics(), orient='index') if new else None
        src = 'ACS 2019-2023 B19013 + B03002 (recovered; same source and code as the release)'
    rows = []
    if new:
        dem.index = dem.index.astype('int64')
        check(lk.set_index('block_group_id'), inc, dem)
        for i in new:
            v = inc.get(i, np.nan)
            d = dem.loc[i] if i in dem.index else pd.Series(dtype=float)
            rows.append(dict(block_group_id=i, median_household_income=v, has_income='yes' if pd.notna(v) else 'no',
                             **{a: d.get(a, np.nan) for a in ATTR[2:]}, n_trees_orig=0, attribute_source=src))
        lk = pd.concat([lk, pd.DataFrame(rows)], ignore_index=True)
    print(f'{city}: {len(ids)} units in boundary file; {len(lk) - len(rows)} original; {len(rows)} recovered '
          f'({sum(r["has_income"] == "yes" for r in rows)} with income); source verified on all original units' if rows else
          f'{city}: {len(ids)} units, all in the original release')
    lk.to_csv(f'{OUT}/bg_lookup_{city}.csv', index=False)
    prov.append(dict(city=city, units_original=int((lk.attribute_source == 'original release (verbatim)').sum()),
                     units_recovered=int((lk.attribute_source != 'original release (verbatim)').sum())))
pd.DataFrame(prov).to_csv(f'{OUT}/census_attribute_provenance.csv', index=False)
print(pd.DataFrame(prov).to_string(index=False))
