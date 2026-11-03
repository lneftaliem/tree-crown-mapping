"""
manuscript_stats.py -- the tree-derived numbers quoted in the manuscript text, from the census-unit tables and points of
the chosen dataset (same definitions as the figures):
  unadjusted Spearman (income_zscore vs tree_density_per_km2 / canopy_cover_pct_corrected / mean_crown_area_m2 [>= 10
  trees]) with n -- the Figure 2-4 annotations (generate_plots_corrected.py computes the identical statistics);
  city totals: trees, mean / median crown area (all crowns), total summed crown area and its share of municipal land
  (citywide summed-crown canopy), census units with detections (from ED Table 1), analysis units, units >= 10 trees,
  median / IQR of unit canopy cover, quartile means of each metric.
Writes analysis_output/stats/{spearman_unadjusted.csv, city_summary.csv, quartile_means.csv}.
"""
import os, sys
import numpy as np, pandas as pd, pyogrio
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'repo_adapted'))
import repro_data  # noqa: E402

B = repro_data.BASE_DIR
OUT = os.path.join(B, 'analysis_output', 'stats')
os.makedirs(OUT, exist_ok=True)
CITIES = ['austin', 'bloomington', 'cupertino', 'surrey']
d = pd.read_csv(os.path.join(repro_data.EQUITY_STAGE_DIR, 'all_cities_urban_covariates_corrected.csv'))
t1 = pd.read_csv(os.path.join(B, 'analysis_output', 'extended_data', 'ED_Table1.csv'))
METRICS = [('tree_density_per_km2', 'tree density'), ('canopy_cover_pct_corrected', 'canopy cover'), ('mean_crown_area_m2', 'mean crown area')]
sp, cs, qm = [], [], []
for i, c in enumerate(CITIES):
    g = d[d.city == c]
    for col, lab in METRICS:
        s = g[g.tree_count >= 10] if col == 'mean_crown_area_m2' else g
        s = s.dropna(subset=[col, 'income_zscore'])
        r = spearmanr(s.income_zscore, s[col])
        sp.append(dict(city=c, metric=lab, n=len(s), rho=r.statistic, p=r.pvalue, significant_05=r.pvalue < 0.05))
    pts = pyogrio.read_dataframe(repro_data.ensure_city_points(c), columns=['crown_area_m2', 'crown_area_px'] if 'crown_area_m2' in
                                 pyogrio.read_info(repro_data.ensure_city_points(c))['fields'] else ['crown_area_px', 'latitude'],
                                 read_geometry=False)
    repro_data.apply_crown_area_correction(pts, c)
    land = float(t1['Municipal land area (km2)'].iloc[i])
    cs.append(dict(city=c, trees=len(pts), mean_crown_m2=pts.crown_area_m2.mean(), median_crown_m2=pts.crown_area_m2.median(),
                   p90_crown_m2=pts.crown_area_m2.quantile(.9), share_crowns_gt100m2=float((pts.crown_area_m2 > 100).mean()),
                   total_crown_area_km2=pts.crown_area_m2.sum() / 1e6, summed_crown_share_of_land_pct=100 * pts.crown_area_m2.sum() / 1e6 / land,
                   units_with_detections=int(t1['Census units with detections'].iloc[i]), analysis_units=len(g),
                   analysis_units_ge10_trees=int((g.tree_count >= 10).sum()),
                   unit_canopy_median_pct=g.canopy_cover_pct_corrected.median(), unit_canopy_q25=g.canopy_cover_pct_corrected.quantile(.25),
                   unit_canopy_q75=g.canopy_cover_pct_corrected.quantile(.75), unit_density_median=g.tree_density_per_km2.median()))
    for q, gq in g.groupby('income_quartile'):
        qm.append(dict(city=c, income_quartile=q, n=len(gq), mean_density=gq.tree_density_per_km2.mean(),
                       mean_canopy=gq.canopy_cover_pct_corrected.mean(), mean_crown=gq.mean_crown_area_m2.mean()))
pd.DataFrame(sp).to_csv(f'{OUT}/spearman_unadjusted.csv', index=False)
pd.DataFrame(cs).to_csv(f'{OUT}/city_summary.csv', index=False)
pd.DataFrame(qm).to_csv(f'{OUT}/quartile_means.csv', index=False)
pd.set_option('display.width', 220)
print(pd.DataFrame(sp).round(4).to_string(index=False)); print(pd.DataFrame(cs).round(3).to_string(index=False))
