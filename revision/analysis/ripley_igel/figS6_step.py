"""
figS6_step.py -- the figS6 step of run_analysis.py: Fig. S6 and the Ripley's L statistics of Supplementary Note 5 with
Christian Igel's code (process_city_trees.py + his notebook analysis, github.com/christian-igel/city-trees @ 3c57d4c),
in the UTM zone of each city, with the convex hull of the trees as the area.

Stages a work directory <run>/analysis_output/ripley_igel/ with the dataset's tree points and the census-unit
boundaries (through repro_data, so they are recorded in provenance/inputs_used.csv), runs run_ripley.py
(RIPLEY_PROJ=utm) and analyze_notebook.py (FIG_DPI=300), and writes a per-census-area table without geometry.
Outputs: notebook_stats.csv, notebook_quartiles.csv, ripley_by_unit.csv, L_quartiles_paper.{png,pdf} (= Fig. S6),
<city>_Ripley.parquet.
"""
import os, sys, subprocess
import numpy as np, pandas as pd, geopandas as gpd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'repo_adapted'))
import repro_data  # noqa: E402

CITIES = ['austin', 'bloomington', 'cupertino', 'surrey']
W = os.path.join(repro_data.BASE_DIR, 'analysis_output', 'ripley_igel')
os.makedirs(f'{W}/census_boundaries', exist_ok=True)
for c in CITIES:
    pts = repro_data.ensure_city_points(c)
    _, bg = repro_data.ensure_boundaries(c)
    for src, dst in ((pts, f'{W}/{c}_ripley_points_with_income.geojson'),
                     (bg, f'{W}/census_boundaries/{c}_exact_block_groups.geojson')):
        if not os.path.lexists(dst):
            os.symlink(os.path.realpath(src), dst)
env = dict(os.environ, RIPLEY_PROJ='utm', RIPLEY_AREA='hull', FIG_DPI='300')
subprocess.run([sys.executable, f'{HERE}/run_ripley.py', W] + CITIES, env=env, check=True)
subprocess.run([sys.executable, f'{HERE}/analyze_notebook.py', W, 'final'], env=env, check=True)
rows = []
for c in CITIES:
    g = gpd.read_parquet(f'{W}/{c}_Ripley.parquet')
    idc = 'DAUID' if 'DAUID' in g.columns else 'GEOID'
    L = np.vstack([np.asarray(v, float) for v in g['L']])
    t = pd.DataFrame({'city': c, 'unit': g[idc].astype('int64'), 'has_income': g['has_income'], 'income': g['income'],
                      'trees': g['trees'], 'processed': g['processed'], 'processed_area_m2': g['processed area'],
                      'auc': g['auc']})
    rows.append(pd.concat([t, pd.DataFrame(L, columns=[f'L_{int(r)}' for r in np.linspace(2, 50, 25)])], axis=1))
pd.concat(rows).to_csv(f'{W}/ripley_by_unit.csv', index=False)
print(open(f'{W}/notebook_stats.csv').read())
