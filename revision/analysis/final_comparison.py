"""
final_comparison.py -- published manuscript numbers vs the corrected (v1b) results, with change flags.

Reads only analysis_output tables of guarded runs (run_analysis.py) and the hand-transcribed published values
(outputs/final_v1b/published_values.csv; source = manuscript_original text). No tree data are read.
Runs compared (label=dir): the published-data reproduction, the primary corrected run, and its sensitivities.
For every income test the change from PUBLISHED to PRIMARY is classified:
  direction     sign of rho / beta differs
  significance  p < 0.05 on one side only (adjusted models: also BH q < 0.05)
  magnitude     |delta rho| >= 0.10, or |beta| ratio outside [0.5, 2]
Usage: python final_comparison.py <out_dir> published=<csv> primary=<run> [label=<run> ...]
Writes <out_dir>/{income_tests_comparison.csv, descriptive_comparison.csv, other_comparison.csv}.
"""
import os, sys
import numpy as np, pandas as pd

OUT = sys.argv[1]
args = dict(a.split('=', 1) for a in sys.argv[2:])
PUB = pd.read_csv(args.pop('published'))
RUNS = args  # primary first, then sensitivities / reproduction
os.makedirs(OUT, exist_ok=True)
CITIES = ['austin', 'bloomington', 'cupertino', 'surrey']
A = lambda run, rel: pd.read_csv(f'{RUNS[run]}/analysis_output/{rel}')


def pnum(x):
    try:
        return float(str(x).lstrip('<>'))
    except ValueError:
        return np.nan


def flags(pr, pp, r, p, beta=False):
    f = []
    if np.sign(pr) != np.sign(r) and not (abs(pr) < 1e-12):
        f.append('direction')
    if (pp < 0.05) != (p < 0.05):
        f.append('significance')
    if beta:
        ratio = abs(r) / abs(pr) if pr else np.inf
        if ratio < 0.5 or ratio > 2:
            f.append('magnitude')
    elif abs(r - pr) >= 0.10:
        f.append('magnitude')
    return ';'.join(f) or 'unchanged'


rows = []
# unadjusted Spearman
pub = PUB[PUB.quantity == 'spearman_rho']
for _, q in pub.iterrows():
    row = dict(test='Spearman (unadjusted)', city=q.city, metric=q.metric, published=float(q.published), published_p=q.published_p)
    for lab in RUNS:
        s = A(lab, 'stats/spearman_unadjusted.csv').set_index(['city', 'metric'])
        if (q.city, q.metric) in s.index:
            t = s.loc[(q.city, q.metric)]
            row.update({f'rho|{lab}': t.rho, f'p|{lab}': t.p, f'n|{lab}': t.n})
    rows.append(row)
# adjusted OLS (HC3) + spatial error models; published values exist only for the 3 nominally significant tests,
# the other 9 were stated non-significant (p >= 0.05) and none survived BH
pa = PUB[PUB.quantity == 'ols_beta'].set_index(['city', 'metric'])
for c in CITIES:
    for o in ('Tree Density', 'Canopy Cover', 'Crown Area'):
        row = dict(test='OLS HC3 (adjusted)', city=c, metric=o)
        if (c, o) in pa.index:
            row.update(published=float(pa.loc[(c, o), 'published']), published_p=pa.loc[(c, o), 'published_p'])
        else:
            row.update(published=np.nan, published_p='>=0.05 (not reported)')
        for lab in RUNS:
            s = A(lab, 'adjusted_models/ols_hc3.csv').set_index(['city', 'outcome'])
            e = A(lab, 'adjusted_models/spatial_error.csv').set_index(['city', 'outcome'])
            if (c, o) in s.index:
                t, u = s.loc[(c, o)], e.loc[(c, o)]
                row.update({f'beta|{lab}': t.beta, f'p|{lab}': t.pvalue, f'q|{lab}': t.bh_qvalue, f'n|{lab}': t.N,
                            f'queen_beta|{lab}': u.queen_beta, f'queen_p|{lab}': u.queen_p,
                            f'knn5_beta|{lab}': u.knn5_beta, f'knn5_p|{lab}': u.knn5_p})
        rows.append(row)
T = pd.DataFrame(rows)
prim = list(RUNS)[0]
ch = []
for _, r in T.iterrows():
    if r.test.startswith('Spearman'):
        ch.append(flags(r.published, pnum(r.published_p), r[f'rho|{prim}'], r[f'p|{prim}']))
    else:
        pp = pnum(r.published_p)
        b = r[f'beta|{prim}']
        if np.isnan(r.published):  # published: non-significant, BH non-significant
            f = []
            if r[f'p|{prim}'] < 0.05:
                f.append('significance')
            if r[f'q|{prim}'] < 0.05:
                f.append('BH-significant (published: none)')
            ch.append(';'.join(f) or 'unchanged (non-significant)')
        else:
            f = flags(r.published, pp, b, r[f'p|{prim}'], beta=True)
            if r[f'q|{prim}'] < 0.05:
                f = (f if f != 'unchanged' else '') + (';' if f != 'unchanged' else '') + 'BH-significant (published: none)'
            ch.append(f)
T.insert(5, 'change_published_to_primary', ch)
T.to_csv(f'{OUT}/income_tests_comparison.csv', index=False)

# descriptive numbers
d = []
for lab in RUNS:
    cs = A(lab, 'stats/city_summary.csv').set_index('city')
    e = A(lab, 'extended_data/ED_Table1.csv')
    e.index = CITIES
    for c in CITIES:
        d.append(dict(run=lab, city=c, trees=cs.loc[c, 'trees'], mean_crown_m2=cs.loc[c, 'mean_crown_m2'],
                      median_crown_m2=cs.loc[c, 'median_crown_m2'], share_crowns_gt100m2=cs.loc[c, 'share_crowns_gt100m2'],
                      summed_crown_share_of_land_pct=cs.loc[c, 'summed_crown_share_of_land_pct'],
                      units_with_detections=cs.loc[c, 'units_with_detections'], analysis_units=cs.loc[c, 'analysis_units'],
                      analysis_units_ge10=cs.loc[c, 'analysis_units_ge10_trees'],
                      median_density_ed_table1=e.loc[c, 'Median density (trees km-2)'],
                      max_density_ed_table1=e.loc[c, 'Maximum density (trees km-2)']))
D = pd.DataFrame(d)
pubd = PUB[PUB.quantity.isin(['trees', 'mean_crown_m2', 'units_with_detections', 'median_density', 'max_density',
                              'analysis_units', 'analysis_units_ge10'])]
m = {'trees': 'trees', 'mean_crown_m2': 'mean_crown_m2', 'units_with_detections': 'units_with_detections',
     'median_density': 'median_density_ed_table1', 'max_density': 'max_density_ed_table1', 'analysis_units': 'analysis_units',
     'analysis_units_ge10': 'analysis_units_ge10'}
pp = pubd.assign(col=pubd.quantity.map(m)).pivot_table(index='city', columns='col', values='published', aggfunc='first')
pp = pp.apply(pd.to_numeric).reset_index().assign(run='published (manuscript text)')
pd.concat([pp, D], ignore_index=True)[['run'] + [c for c in D.columns if c != 'run']].sort_values(['city', 'run']).to_csv(
    f'{OUT}/descriptive_comparison.csv', index=False)

# other tree-derived analyses
o = []
has = lambda lab, rel: os.path.exists(f'{RUNS[lab]}/analysis_output/{rel}')
for lab in RUNS:
    rp = A(lab, 'ripley/ripley_auc_income_spearman.csv')
    for _, r in rp.iterrows():
        o.append(dict(run=lab, analysis='Ripley AUC vs income (Spearman)', city=r.city, n=r.n_census_areas,
                      value=r.spearman_rho_auc_income, p=r.p_value, extra=f'share clustered all r = {r.share_units_clustered_all_r:.3f}'))
    if not has(lab, 'inventory_correlation/inventory_correlation_stats.csv'):
        continue  # partial sensitivity run (socioeconomic steps only)
    iv = A(lab, 'inventory_correlation/inventory_correlation_stats.csv')
    for _, r in iv.iterrows():
        o.append(dict(run=lab, analysis='Inventory vs model counts (Spearman)', city=r.city, n=r.n_units, value=r.spearman_rho,
                      p=r.p_value, extra=f'model trees in units = {r.model_trees_in_units:.0f}'))
    h = A(lab, 'redlining_figures/austin_holc_grade_metrics.csv')
    for _, r in h.iterrows():
        o.append(dict(run=lab, analysis=f'HOLC grade {r.grade}', city='austin', n=r.total_trees, value=r.weighted_tree_density_km2,
                      p=np.nan, extra=f'mean crown {r.weighted_mean_crown_m2:.3f} m2; crown fraction {r.weighted_crown_fraction_pct:.3f} %'))
    s = A(lab, 'surrey_reviewer2_analysis/surrey_sensitivity_scenarios_summary.csv')
    for _, r in s.iterrows():
        o.append(dict(run=lab, analysis=f'Fig S12: {r.Scenario}', city='surrey', n=r.Retained_DAs, value=r.Spearman_Rho,
                      p=r.P_Value, extra=f'excluded area {r.Excluded_Area_Pct:.1f} %'))
pd.DataFrame(o).to_csv(f'{OUT}/other_comparison.csv', index=False)
pd.set_option('display.width', 250)
print(T[['test', 'city', 'metric', 'published', 'published_p', 'change_published_to_primary']].to_string(index=False))
