"""
17b_crown_ndvi_join.py -- joins the per-label crown NDVI of 17_crown_ndvi.py to the final dataset crowns (source_tile,
instance_label) and summarizes the share of crowns / crown area below NDVI 0, 0.1, 0.2, 0.3 per city.
Writes <work>/diag/crown_ndvi_final_<city>.parquet and <work>/diag/crown_ndvi_summary.csv.
Usage: python 17b_crown_ndvi_join.py <work_dir> <dataset_dir>
"""
import sys, glob
import pandas as pd, pyogrio

work, ds = sys.argv[1:3]
CITIES = ('austin', 'bloomington', 'cupertino', 'surrey')
nd = pd.concat(pd.read_parquet(f) for f in glob.glob(f'{work}/diag/crown_ndvi/*.parquet'))
rows = []
for c in CITIES:
    p = pyogrio.read_dataframe(f'{ds}/{c}_ripley_points_with_income.geojson', read_geometry=False,
                               columns=['source_tile', 'instance_label', 'crown_area_m2', 'block_group_id', 'median_household_income'])
    m = p.merge(nd, left_on=['source_tile', 'instance_label'], right_on=['tile', 'label'], how='left')
    a = m.crown_area_m2
    r = dict(city=c, n=len(m), matched=m.ndvi_mean.notna().mean(), ndvi_median=m.ndvi_mean.median())
    for t in (0, 0.1, 0.2, 0.3):
        s = m.ndvi_mean < t
        r[f'n_lt{t}'] = s.mean(); r[f'area_lt{t}'] = a[s].sum() / a.sum()
    rows.append(r)
    m[['source_tile', 'instance_label', 'block_group_id', 'crown_area_m2', 'ndvi_mean', 'share_ndvi_lt01']].to_parquet(
        f'{work}/diag/crown_ndvi_final_{c}.parquet')
out = pd.DataFrame(rows); out.to_csv(f'{work}/diag/crown_ndvi_summary.csv', index=False)
pd.set_option('display.width', 250); print(out.round(3).to_string(index=False))
