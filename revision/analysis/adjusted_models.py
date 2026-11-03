"""
adjusted_models.py -- park-adjusted OLS (HC3) and spatial error models (Extended Data Figs 3 and 4).

The model code is the 'original specification' branch of the authoritative script
sherlock_venv/boundary_denominator_sensitivity/run_park_adjusted_sensitivity_models.py (it reproduces the published
Extended Data Fig. 4 p-values), restricted to the published specification and fed with the census-unit tables built by
build_unit_table.py for the chosen dataset:
  outcome      log10(y + 1) of tree_density_per_km2, canopy_cover_pct_corrected, mean_crown_area_m2 (units with >= 10 trees)
  predictors   income_zscore, housing_density_log10, dist_cbd_km, park_pct (+ intercept)
  OLS          statsmodels OLS, cov_type='HC3'; 95 % CI with z = 1.96 (statsmodels conf_int on a normal-based HC3 fit)
  spatial      spreg.ML_Error with row-standardised Queen contiguity (islands joined to their 2 nearest neighbours)
               and 5-nearest-neighbour weights, on municipality-clipped unit geometry in the city UTM zone
  Moran's I    on plain OLS residuals, 9,999 permutations (Supplementary Note 3)
  multiplicity Benjamini-Hochberg across the 12 OLS income tests
Writes $TREE_MAPPING_BASE_DIR/analysis_output/adjusted_models/{ols_hc3.csv, spatial_error.csv, residual_moran.csv}.
"""
import os, sys
import numpy as np, pandas as pd, geopandas as gpd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import libpysal
from spreg import ML_Error
from esda.moran import Moran

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'repo_adapted'))
import repro_data  # noqa: E402

OUT = os.path.join(repro_data.BASE_DIR, 'analysis_output', 'adjusted_models')
os.makedirs(OUT, exist_ok=True)
UTM = {'austin': 32614, 'bloomington': 32616, 'cupertino': 32610, 'surrey': 32610}
CITIES = ['austin', 'bloomington', 'cupertino', 'surrey']
LABEL = {'austin': 'Austin, TX', 'bloomington': 'Bloomington, IN', 'cupertino': 'Cupertino, CA', 'surrey': 'Surrey, BC'}
OUTCOMES = [('Tree Density', 'tree_density_per_km2'), ('Canopy Cover', 'canopy_cover_pct_corrected'),
            ('Crown Area', 'mean_crown_area_m2')]
X_VARS = ['income_zscore', 'housing_density_log10', 'dist_cbd_km', 'park_pct']
PERM = int(os.environ.get('MORAN_PERMUTATIONS', '9999'))

repro_data.stage_covariates()
cov = pd.read_csv(os.path.join(repro_data.EQUITY_STAGE_DIR, 'all_cities_urban_covariates_corrected.csv'))
park = pd.read_csv(os.path.join(repro_data.EQUITY_STAGE_DIR, 'all_cities_park_covariates.csv'))
df = cov.merge(park[['city', 'GEOID', 'park_pct']], on=['city', 'GEOID'], how='inner')
assert len(df) == len(cov)

ols, spat, mor = [], [], []
for city in CITIES:
    repro_data.ensure_boundaries(city)
    bg = gpd.read_file(os.path.join(repro_data.BOUNDARIES_STAGE_DIR, f'{city}_exact_block_groups.geojson'))
    bg['GEOID'] = bg['DAUID' if city == 'surrey' else 'GEOID'].astype('int64')
    m = bg[['GEOID', 'geometry']].merge(df[df.city == city], on='GEOID', how='inner').to_crs(UTM[city])
    assert len(m) == (df.city == city).sum(), f'{city}: geometry missing for some units'
    for out_name, col in OUTCOMES:
        sub = m.dropna(subset=[col] + X_VARS)
        if out_name == 'Crown Area':
            sub = sub[sub.tree_count >= 10]
        sub = sub.reset_index(drop=True)
        N = len(sub)
        y = np.log10(sub[col].values + 1.0)
        X = sub[X_VARS].values
        fit = sm.OLS(y, sm.add_constant(X)).fit(cov_type='HC3')
        ci = fit.conf_int()[1]
        ols.append(dict(city=city, city_label=LABEL[city], outcome=out_name, N=N, beta=fit.params[1], se_hc3=fit.bse[1],
                        ci_low=ci[0], ci_high=ci[1], pvalue=fit.pvalues[1], r2_adj=fit.rsquared_adj,
                        **{f'beta_{v}': fit.params[i + 1] for i, v in enumerate(X_VARS[1:], 1)},
                        **{f'p_{v}': fit.pvalues[i + 1] for i, v in enumerate(X_VARS[1:], 1)}))
        wq = libpysal.weights.Queen.from_dataframe(sub, use_index=False)
        if wq.islands:
            kn = libpysal.weights.KNN.from_dataframe(sub, k=2)
            nb = wq.neighbors.copy()
            for i in wq.islands:
                nb[i] = kn.neighbors[i]
            wq = libpysal.weights.W(nb)
        wq.transform = 'R'
        wk = libpysal.weights.KNN.from_dataframe(sub, k=min(5, N - 1)); wk.transform = 'R'
        res = sm.OLS(y, sm.add_constant(X)).fit().resid
        mq, mk = Moran(res, wq, permutations=PERM), Moran(res, wk, permutations=PERM)
        row = dict(city=city, city_label=LABEL[city], outcome=out_name, N=N, ols_beta=fit.params[1],
                   ols_ci_low=ci[0], ols_ci_high=ci[1], ols_p=fit.pvalues[1])
        for tag, w in (('queen', wq), ('knn5', wk)):
            ml = ML_Error(y.reshape(-1, 1), X, w=w, name_x=X_VARS)
            b, se, p = ml.betas[1][0], ml.std_err[1], ml.z_stat[1][1]
            row.update({f'{tag}_beta': b, f'{tag}_se': se, f'{tag}_ci_low': b - 1.95996 * se, f'{tag}_ci_high': b + 1.95996 * se,
                        f'{tag}_p': p, f'{tag}_lambda': ml.betas[-1][0]})
            mr = Moran(ml.e_filtered.flatten(), w, permutations=PERM)
            mor.append(dict(city=city, outcome=out_name, weights=tag, stage='spatially filtered residuals', I=mr.I, p_sim=mr.p_sim))
        spat.append(row)
        mor += [dict(city=city, outcome=out_name, weights='queen', stage='OLS residuals', I=mq.I, p_sim=mq.p_sim),
                dict(city=city, outcome=out_name, weights='knn5', stage='OLS residuals', I=mk.I, p_sim=mk.p_sim)]
O = pd.DataFrame(ols)
rej, q, _, _ = multipletests(O.pvalue, alpha=0.05, method='fdr_bh')
O['bh_qvalue'], O['bh_significant'] = q, rej
O.to_csv(f'{OUT}/ols_hc3.csv', index=False)
pd.DataFrame(spat).to_csv(f'{OUT}/spatial_error.csv', index=False)
pd.DataFrame(mor).to_csv(f'{OUT}/residual_moran.csv', index=False)
pd.set_option('display.width', 200)
print(O[['city', 'outcome', 'N', 'beta', 'ci_low', 'ci_high', 'pvalue', 'bh_qvalue']].round(4).to_string(index=False))
print(pd.DataFrame(spat)[['city', 'outcome', 'queen_beta', 'queen_p', 'knn5_beta', 'knn5_p']].round(4).to_string(index=False))
