"""
analyze_notebook.py -- cells 2 and 4 of Christian Igel's notebook "City Trees Analysis.ipynb"
(github.com/christian-igel/city-trees, commit 3c57d4c), with the same logic, run on the <city>_Ripley.parquet files of
one work directory (written by run_ripley.py):
  filter: has_income & processed >= 50 (trees in the polygons that were analysed)
  Spearman(auc, income)
  quartile = pd.qcut(income, 4); L averaged within quartile weighted by 'processed area' (convex-hull area)
  figure: 2 x 2, shared y, ax[k % 2, k // 2] for cities [austin, cupertino, bloomington, surrey], legend 'quartile'
Additions that do not change the computation: the numbers are also written to CSV, LaTeX text rendering is off, and
FIG_DPI sets the PNG resolution (default 100 = the notebook's; 300 for the manuscript figure).
Usage: [FIG_DPI=300] python analyze_notebook.py <work dir> <label>
Writes <work dir>/notebook_stats.csv, notebook_quartiles.csv, L_quartiles_paper.{png,pdf}.
"""
import sys, os
import numpy as np, pandas as pd, geopandas as gpd
from scipy.stats import spearmanr
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

work, label = sys.argv[1], sys.argv[2]
os.chdir(work)
cities = ["austin", "cupertino", "bloomington", "surrey"]
min_trees = 50
weighted = True

LQs, stats, qrows = [], [], []
for k, city in enumerate(cities):
    print(f"Processing {city.title()}")
    gdf = gpd.read_parquet(f"{city}_Ripley.parquet")
    print(f"\tTotal number of areas: {len(gdf)}")
    df_filtered = gdf[(gdf["has_income"]) & (gdf["processed"] >= min_trees)]
    print(f"\tNumber of areas with more than {min_trees} trees and income: {len(df_filtered)}")
    corr, pval = spearmanr(df_filtered["auc"], df_filtered["income"])
    print(f"\tRank correlation between AUC and income: {corr:.2}  (p={pval:.2})")
    df_filtered = df_filtered.copy()
    df_filtered["quartile"] = pd.qcut(df_filtered["income"], q=4, labels=False) + 1
    if weighted == False:
        LQ = df_filtered.groupby("quartile")["L"].apply(lambda x: np.mean(np.stack(x), axis=0))
    else:
        LQ = df_filtered.groupby("quartile").apply(
            lambda x: np.average(np.stack(x["L"].values), axis=0, weights=x["processed area"].values))
    LQs.append(LQ)
    stats.append(dict(data=label, city=city, total_areas=len(gdf), n_areas=len(df_filtered), rho=corr, p=pval))
    sup = np.linspace(2, 50, 25)
    for q, L in LQ.items():
        qrows.append(dict(data=label, city=city, quartile=q, n=int((df_filtered.quartile == q).sum()),
                          **{f'L_{int(r)}': v for r, v in zip(sup, L)}))

pd.DataFrame(stats).to_csv('notebook_stats.csv', index=False)
pd.DataFrame(qrows).to_csv('notebook_quartiles.csv', index=False)

plt.rcParams.update({"text.usetex": False, "font.family": "Arial", "axes.titlesize": 20, "axes.titleweight": "bold",
                     "axes.labelsize": 10, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.2})
fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharey=True)
for k, city in enumerate(cities):
    sup = np.linspace(2, 50, 25)
    the_ax = ax[k % 2, k // 2]
    for quartile, L in LQs[k].items():
        the_ax.plot(sup, L, label=quartile)
    the_ax.set_title(city.title())
    the_ax.grid()
    if k == 0:
        the_ax.legend(title="quartile")
    if k // 2 == 0:
        the_ax.set_ylabel(r"$\widehat{L}(r)$")
    if k % 2 == 1:
        the_ax.set_xlabel(r"$r$")
plt.savefig("L_quartiles_paper.pdf", bbox_inches='tight')
plt.savefig("L_quartiles_paper.png", bbox_inches="tight", dpi=int(os.environ.get("FIG_DPI", "100")))  # dpi: addition; 100 = matplotlib default
