"""
ed_fig1.py -- Extended Data Fig. 1: unique trees retained (%) at centroid-deduplication radii 0.8, 1.0, 1.5 m.

Counts come from the dataset build itself (code/pipeline/06_build_city_outputs.py records, for every city, the number of
final in-city crowns when its cross-tile dedup is run at 0.8 / 1.0 / 1.5 m with every other step unchanged; the 1.0 m
count equals the delivered dataset). Path of the build log JSON(s): $DATASET_BUILD_LOGS (glob).
The plotting code is the original run_sensitivity.py plotting block, unchanged except for the input table and output
path. (The original figure re-detected trees from rasters and deduplicated in degrees, 1 m ~ 1e-5 deg; the v1 build
deduplicates in projected metres with edge-distance priority, so v1 values are computed with the final method.)
"""
import os, glob, json
from math import log10, floor
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

BASE = os.environ['TREE_MAPPING_BASE_DIR']
OUT = os.path.join(BASE, 'analysis_output', 'extended_data')
os.makedirs(OUT, exist_ok=True)
CITIES = ['austin', 'bloomington', 'cupertino', 'surrey']
THRESHOLDS_M = [0.8, 1.0, 1.5]
OI_COLORS = {'austin': '#E69F00', 'bloomington': '#56B4E9', 'cupertino': '#009E73', 'surrey': '#CC79A7'}
CITY_LABELS = {'austin': 'Austin, TX', 'bloomington': 'Bloomington, IN', 'cupertino': 'Cupertino, CA', 'surrey': 'Surrey, BC'}

logs = {}
for f in glob.glob(os.environ['DATASET_BUILD_LOGS']):
    d = json.load(open(f))
    logs.update(d if all(k in CITIES for k in d) else {})
rows = []
for c in CITIES:
    for r in THRESHOLDS_M:
        rows.append(dict(city=c.title(), threshold_m=r, n_trees=logs[c][f'dedup_sensitivity_final_r{r}'],
                         removed=logs[c][f'dedup_sensitivity_removed_r{r}']))
    assert logs[c]['dedup_sensitivity_final_r1.0'] == logs[c]['final'], c
df_res = pd.DataFrame(rows)
df_res['pct_retained'] = df_res.n_trees / df_res.groupby('city').n_trees.transform(lambda s: s.iloc[1]) * 100
df_res.to_csv(f'{OUT}/ED_Fig1_dedup_sensitivity.csv', index=False)

all_pcounts = df_res.pct_retained.tolist()
y_min, y_max = min(all_pcounts) - 0.2, max(all_pcounts) + 0.2
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()
for i, city in enumerate(CITIES):  # ---- original plotting block (run_sensitivity.py) ----
    ax = axes[i]
    city_df = df_res[df_res['city'] == city.title()].copy()
    count_10 = city_df[city_df['threshold_m'] == 1.0]['n_trees'].values[0]
    city_df['pct_retained'] = city_df['n_trees'] / count_10 * 100
    ax.plot(city_df['threshold_m'], city_df['pct_retained'], marker='o', linestyle='-', color=OI_COLORS[city], linewidth=2, markersize=8)
    ax.set_title(CITY_LABELS[city], fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Deduplication Threshold (meters)', fontsize=10)
    ax.set_ylabel('Unique Trees Retained (%)', fontsize=10)
    ax.set_xticks(THRESHOLDS_M)
    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=2))
    ax.ticklabel_format(useOffset=False, style='plain', axis='x')
    ax.grid(True, linestyle='--', alpha=0.7)
    rounded_n = int(round(count_10, -floor(log10(count_10)) + 2)) if count_10 > 0 else 0
    ax.text(0.95, 0.95, f"Standard N = {rounded_n:,} at 1.0 m", transform=ax.transAxes, verticalalignment='top',
            horizontalalignment='right', color='dimgray', fontsize=9, fontweight='semibold')
plt.tight_layout()
for ext in ('png', 'pdf', 'svg'):
    plt.savefig(f'{OUT}/ED_Fig1_dedup_sensitivity.{ext}', dpi=300)
print(df_res.round(3).to_string(index=False))
