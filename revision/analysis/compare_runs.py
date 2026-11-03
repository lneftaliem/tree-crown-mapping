"""
compare_runs.py -- side-by-side tables of the tree-derived results of several guarded analysis runs (run_analysis.py),
e.g. original unsplit / mixed-vintage split / final v1. Reads only the runs' analysis_output tables (no tree data).
Usage: python compare_runs.py <out_dir> label=<run dir> [label=<run dir> ...]
Writes <out_dir>/{compare_city_summary,compare_spearman,compare_adjusted,compare_ripley,compare_sensitivity}.csv
"""
import os, sys
import pandas as pd

OUT = sys.argv[1]
RUNS = dict(a.split('=', 1) for a in sys.argv[2:])
os.makedirs(OUT, exist_ok=True)


def gather(rel, keys, cols):
    parts = []
    for lab, d in RUNS.items():
        p = f'{d}/analysis_output/{rel}'
        if not os.path.exists(p):
            print('missing', lab, rel); continue
        t = pd.read_csv(p)
        parts.append(t[keys + [c for c in cols if c in t.columns]].assign(run=lab))
    t = pd.concat(parts)
    w = t.pivot_table(index=keys, columns='run', values=[c for c in cols if c in t.columns], aggfunc='first')
    w = w.reindex(columns=[(c, r) for c in cols for r in RUNS if (c, r) in w.columns])
    w.columns = [f'{c}|{r}' for c, r in w.columns]
    return w.reset_index()


t1 = []
for lab, d in RUNS.items():
    cs = pd.read_csv(f'{d}/analysis_output/stats/city_summary.csv')
    e = pd.read_csv(f'{d}/analysis_output/extended_data/ED_Table1.csv')
    cs['median_density_fig1_rule'] = e['Median density (trees km-2)'].values
    cs['max_density_fig1_rule'] = e['Maximum density (trees km-2)'].values
    t1.append(cs.assign(run=lab))
cs = pd.concat(t1).set_index(['city', 'run']).T
cs = cs.reindex(columns=pd.MultiIndex.from_product([sorted({c for c, _ in cs.columns}, key=['austin', 'bloomington', 'cupertino', 'surrey'].index), list(RUNS)]))
cs.to_csv(f'{OUT}/compare_city_summary.csv')
gather('stats/spearman_unadjusted.csv', ['city', 'metric'], ['n', 'rho', 'p']).to_csv(f'{OUT}/compare_spearman.csv', index=False)
gather('adjusted_models/ols_hc3.csv', ['city', 'outcome'], ['N', 'beta', 'pvalue', 'bh_qvalue']).merge(
    gather('adjusted_models/spatial_error.csv', ['city', 'outcome'], ['queen_beta', 'queen_p', 'knn5_beta', 'knn5_p']),
    on=['city', 'outcome']).to_csv(f'{OUT}/compare_adjusted.csv', index=False)
gather('ripley/ripley_auc_income_spearman.csv', ['city'], ['n_census_areas', 'spearman_rho_auc_income', 'p_value',
       'share_units_clustered_all_r']).to_csv(f'{OUT}/compare_ripley.csv', index=False)
if all(os.path.exists(f'{d}/analysis_output/sensitivity/canopy_denominator_spearman.csv') for d in RUNS.values()):
    gather('sensitivity/canopy_denominator_spearman.csv', ['city', 'metric', 'specification'], ['n', 'rho', 'p']).to_csv(
        f'{OUT}/compare_sensitivity.csv', index=False)
pd.set_option('display.width', 250)
print(pd.read_csv(f'{OUT}/compare_spearman.csv').round(4).to_string(index=False))
