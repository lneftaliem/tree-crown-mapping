"""
surrey_crop_effect.py -- DIAGNOSTIC (does not modify any dataset or run): where the row-crop false positives are and
whether they could move the income findings. Crowns are flagged by the validated row-texture score of their 64 m block
(> 150: ~80 % crop/field on visual validation; > 100: liberal). For each city: flagged share per census unit and its
relation to income; then the primary unit metrics are recomputed WITHOUT flagged crowns (same formulas as
build_unit_table.py: density capped at 30,000, summed crown area / full unit area, mean crown area for >= 10 trees,
same analysis units and covariates) and the unadjusted Spearman and OLS-HC3 income tests are repeated.
Writes outputs/landcover_review/crop_effect_{units,tests}.csv.
"""
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy.stats import spearmanr
P = '/path/to/Urban_data/tree_crowns_v1_20261101'
U = pd.read_csv(f'{P}/outputs/runs/v1b_final/analysis_output/equity_results/all_cities_urban_covariates_corrected.csv')
PK = pd.read_csv(f'{P}/outputs/runs/v1b_final/analysis_output/equity_results/all_cities_park_covariates.csv')
U = U.merge(PK[['city', 'GEOID', 'park_pct']], on=['city', 'GEOID'])
X = ['income_zscore', 'housing_density_log10', 'dist_cbd_km', 'park_pct']
units, tests = [], []
for c in ['austin', 'bloomington', 'cupertino', 'surrey']:
    m = pd.read_parquet(f'{P}/work_v1b/diag/row_texture_crowns_{c}.parquet')
    m['median_household_income'] = pd.to_numeric(m.median_household_income, errors='coerce')
    m = m.dropna(subset=['block_group_id', 'median_household_income']); m['GEOID'] = m.block_group_id.astype('int64')
    u = U[U.city == c].set_index('GEOID')
    for thr in (None, 150, 100):
        k = m if thr is None else m[~(m.score > thr)]
        a = k.groupby('GEOID').agg(n=('crown_area_m2', 'size'), area=('crown_area_m2', 'sum'), mean=('crown_area_m2', 'mean'))
        a = a.reindex(u.index)
        d = pd.DataFrame(dict(density=np.minimum(a.n.fillna(0) / u.block_area_km2, 30000),
                              canopy=100 * a.area.fillna(0) / (u.block_area_km2 * 1e6),
                              crown=np.where(a.n >= 10, a['mean'], np.nan)), index=u.index)
        d = d.join(u[X + ['median_income']])
        if thr is not None:
            fl = m[m.score > thr].groupby('GEOID').size().reindex(u.index).fillna(0) / m.groupby('GEOID').size().reindex(u.index)
            if thr == 150:
                units.append(pd.DataFrame(dict(city=c, GEOID=u.index, share_flagged150=fl.values, income=u.median_income.values)))
            r = spearmanr(fl.fillna(0), u.median_income)
            tests.append(dict(city=c, spec=f'flag>{thr}', metric='share of unit crowns flagged vs income', n=len(u), rho=r.statistic, p=r.pvalue,
                              crowns_flagged=int((m.score > thr).sum()), share_crowns_flagged=float((m.score > thr).mean())))
        for name, col in [('tree density', 'density'), ('canopy cover', 'canopy'), ('mean crown area', 'crown')]:
            s = d.dropna(subset=[col])
            r = spearmanr(s.income_zscore, s[col])
            ss = s.dropna(subset=X)
            fit = sm.OLS(np.log10(ss[col].values + 1), sm.add_constant(ss[X].values)).fit(cov_type='HC3')
            tests.append(dict(city=c, spec='primary (all crowns)' if thr is None else f'excluding crowns in blocks with texture score > {thr}',
                              metric=name, n=len(s), rho=r.statistic, p=r.pvalue, ols_beta=fit.params[1], ols_p=fit.pvalues[1]))
T = pd.DataFrame(tests); T.to_csv(f'{P}/outputs/landcover_review/crop_effect_tests.csv', index=False)
pd.concat(units).to_csv(f'{P}/outputs/landcover_review/crop_effect_units.csv', index=False)
pd.set_option('display.width', 250); print(T.round(4).to_string(index=False))
