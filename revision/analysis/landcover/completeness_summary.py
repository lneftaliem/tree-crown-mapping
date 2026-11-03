"""
completeness_summary.py -- DIAGNOSTIC summary of completeness.py per analysis unit (read-only):
  green_share      NDVI >= 0.6 share of the unit's in-city imaged area (model-free vegetation proxy; includes lawns)
  closed_share     share in 'closed-canopy' blocks (64 m blocks >= 70 % NDVI >= 0.6)
  det_green        share of NDVI >= 0.6 pixels covered by kept crowns (detection completeness of green cover)
  det_closed       share of closed-canopy pixels covered by kept crowns
and their Spearman correlation with income, next to the primary canopy metric.
Writes outputs/landcover_review/completeness_units.csv and completeness_tests.csv.
"""
import glob, pandas as pd, numpy as np
from scipy.stats import spearmanr
P = '/path/to/Urban_data/tree_crowns_v1_20261101'
c = pd.concat(pd.read_parquet(f) for f in glob.glob(f'{P}/work_v1b/diag/completeness/*.parquet'))
a = c.groupby(['city', 'uid']).sum(numeric_only=True)
a['green_share'] = a.n_ndvi06 / a.n_px; a['closed_share'] = a.n_closed / a.n_px
a['det_green'] = a.n_crown_ndvi06 / a.n_ndvi06; a['det_closed'] = a.n_crown_closed / a.n_closed.replace(0, np.nan)
a['crown_share'] = a.n_crown / a.n_px
U = pd.read_csv(f'{P}/outputs/runs/v1b_final/analysis_output/equity_results/all_cities_urban_covariates_corrected.csv')
m = U.merge(a.reset_index().rename(columns={'uid': 'GEOID'}), on=['city', 'GEOID'], how='left')
m.to_csv(f'{P}/outputs/landcover_review/completeness_units.csv', index=False)
rows = []
for city, g in m.groupby('city'):
    tot = a.loc[city].sum()
    row = dict(city=city, n=len(g), citywide_green_share=tot.n_ndvi06 / tot.n_px, citywide_closed_share=tot.n_closed / tot.n_px,
               citywide_det_green=tot.n_crown_ndvi06 / tot.n_ndvi06, citywide_det_closed=tot.n_crown_closed / tot.n_closed)
    for v in ['green_share', 'closed_share', 'det_green', 'det_closed', 'canopy_cover_pct_corrected']:
        s = g.dropna(subset=[v]); r = spearmanr(s.median_income, s[v])
        row[f'rho_income_{v}'] = r.statistic; row[f'p_{v}'] = r.pvalue
    rows.append(row)
T = pd.DataFrame(rows); T.to_csv(f'{P}/outputs/landcover_review/completeness_tests.csv', index=False)
pd.set_option('display.width', 250); print(T.round(3).T.to_string())
