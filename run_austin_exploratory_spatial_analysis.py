#!/usr/bin/env python3
"""
run_austin_exploratory_spatial_analysis.py

Exploratory spatial and statistical analysis of tree metrics, household income,
distance to downtown, housing density, and park coverage in Austin, TX.
Manuscript: "Mapping urban tree crowns and socioeconomic patterns across four North American cities"
Communications Sustainability.
"""

import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from scipy.stats import spearmanr
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import libpysal
from spreg import GM_Error_Het
from esda.moran import Moran

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns

# Set publication style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.color'] = '#333333'
plt.rcParams['ytick.color'] = '#333333'
plt.rcParams['text.color'] = '#111111'

# Inputs are the committed covariate tables (data/covariates/) and Austin block
# group boundaries (data/boundaries/), staged by repro_data. Outputs go to
# $TREE_MAPPING_BASE_DIR/analysis_output/austin_spatial_analysis/.
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
import repro_data  # noqa: E402

repro_data.stage_covariates()
repro_data.ensure_boundaries("austin")

BASE_DIR = repro_data.BASE_DIR
EQUITY_DIR = repro_data.EQUITY_STAGE_DIR
CENSUS_DIR = repro_data.BOUNDARIES_STAGE_DIR
OUT_DIR = os.path.join(BASE_DIR, "analysis_output", "austin_spatial_analysis")

os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. LOAD AND MERGE AUTHORITATIVE DATASETS
# -----------------------------------------------------------------------------
print("Loading authoritative datasets...")
master_cov_path = os.path.join(EQUITY_DIR, "all_cities_urban_covariates_corrected.csv")
park_cov_path = os.path.join(EQUITY_DIR, "all_cities_park_covariates.csv")
bg_geojson_path = os.path.join(CENSUS_DIR, "austin_exact_block_groups.geojson")
bnd_geojson_path = os.path.join(CENSUS_DIR, "austin_exact_boundary.geojson")

cov_df = pd.read_csv(master_cov_path)
park_df = pd.read_csv(park_cov_path)
gdf_bg = gpd.read_file(bg_geojson_path)
gdf_bnd = gpd.read_file(bnd_geojson_path)

austin_cov = cov_df[cov_df['city'] == 'austin'].copy()
austin_park = park_df[park_df['city'] == 'austin'].copy()

austin_cov['clean_id_str'] = austin_cov['clean_id'].astype(str).str.zfill(12)
austin_park['clean_id_str'] = austin_park['clean_id'].astype(str).str.zfill(12)
gdf_bg['clean_id_str'] = gdf_bg['GEOID'].astype(str).str.zfill(12)

# Merge datasets
merged_df = pd.merge(austin_cov, austin_park[['clean_id_str', 'park_pct', 'natural_area_pct']], on='clean_id_str', how='inner')
merged_gdf = gdf_bg[['clean_id_str', 'geometry']].merge(merged_df, on='clean_id_str', how='inner')

assert len(merged_gdf) == 616, f"Expected 616 records for Austin, got {len(merged_gdf)}"
print(f"Successfully loaded and merged {len(merged_gdf)} Austin census block groups.")

# Downtown Austin reference point (Texas State Capitol: UTM 621154.36, 3349896.43; Lon -97.7404, Lat 30.2747)
CBD_LON, CBD_LAT = -97.740400, 30.274700

# -----------------------------------------------------------------------------
# 2. SPEARMAN CORRELATION ANALYSIS (TASK 2 & 3)
# -----------------------------------------------------------------------------
print("\nCalculating Spearman correlations and FDR corrections...")
corr_vars = [
    ('median_income', 'Median Household Income ($)'),
    ('dist_cbd_km', 'Distance to Downtown (km)'),
    ('housing_density_km2', 'Housing Density (units/km²)'),
    ('park_pct', 'Park Coverage (%)'),
    ('tree_density_per_km2', 'Tree Density (trees/km²)'),
    ('canopy_cover_pct_corrected', 'Canopy Cover (%)'),
    ('mean_crown_area_m2', 'Mean Crown Area (m²)')
]

var_keys = [v[0] for v in corr_vars]
var_labels = [v[1] for v in corr_vars]

n_vars = len(var_keys)
corr_matrix = np.zeros((n_vars, n_vars))
pval_matrix = np.zeros((n_vars, n_vars))
n_matrix = np.zeros((n_vars, n_vars), dtype=int)

p_list = []
pair_indices = []

for i in range(n_vars):
    for j in range(n_vars):
        v1, v2 = var_keys[i], var_keys[j]
        sub = merged_df.dropna(subset=[v1, v2])
        r, p = spearmanr(sub[v1], sub[v2])
        corr_matrix[i, j] = r
        pval_matrix[i, j] = p
        n_matrix[i, j] = len(sub)
        if i < j:
            p_list.append(p)
            pair_indices.append((i, j))

# FDR correction (Benjamini-Hochberg)
reject, pvals_corrected, _, _ = multipletests(p_list, alpha=0.05, method='fdr_bh')
p_fdr_matrix = np.copy(pval_matrix)
for idx, (i, j) in enumerate(pair_indices):
    p_fdr_matrix[i, j] = pvals_corrected[idx]
    p_fdr_matrix[j, i] = pvals_corrected[idx]

corr_table_df = pd.DataFrame(corr_matrix, index=var_labels, columns=var_labels)
pval_table_df = pd.DataFrame(pval_matrix, index=var_labels, columns=var_labels)
pfdr_table_df = pd.DataFrame(p_fdr_matrix, index=var_labels, columns=var_labels)

corr_table_path = os.path.join(OUT_DIR, "austin_spearman_correlation_matrix.csv")
corr_table_df.to_csv(corr_table_path)
pfdr_table_path = os.path.join(OUT_DIR, "austin_spearman_fdr_pvalues.csv")
pfdr_table_df.to_csv(pfdr_table_path)

# Detailed Pairwise Summary Table
pairwise_rows = []
for idx, (i, j) in enumerate(pair_indices):
    pairwise_rows.append({
        'Variable 1': var_labels[i],
        'Variable 2': var_labels[j],
        'Spearman_rho': corr_matrix[i, j],
        'P_value_raw': pval_matrix[i, j],
        'P_value_FDR': pvals_corrected[idx],
        'Significant_FDR_05': reject[idx],
        'N': n_matrix[i, j]
    })
pairwise_df = pd.DataFrame(pairwise_rows)
pairwise_path = os.path.join(OUT_DIR, "austin_pairwise_spearman_results.csv")
pairwise_df.to_csv(pairwise_path, index=False)
print("Saved Spearman correlation matrices.")

# -----------------------------------------------------------------------------
# 3. QUARTILE COMPARISONS AND SUBGROUP DISTRIBUTIONS (TASK 2 & 3)
# -----------------------------------------------------------------------------
print("\nComputing income and tree metric quartile distributions...")
income_q_summary = merged_df.groupby('income_quartile').agg(
    N=('GEOID', 'count'),
    Income_Mean=('median_income', 'mean'),
    Income_Median=('median_income', 'median'),
    Dist_CBD_Mean=('dist_cbd_km', 'mean'),
    Dist_CBD_Median=('dist_cbd_km', 'median'),
    Dist_CBD_SD=('dist_cbd_km', 'std'),
    Dist_CBD_IQR=('dist_cbd_km', lambda x: x.quantile(0.75) - x.quantile(0.25)),
    Housing_Density_Mean=('housing_density_km2', 'mean'),
    Housing_Density_Median=('housing_density_km2', 'median'),
    Park_Pct_Mean=('park_pct', 'mean'),
    Park_Pct_Median=('park_pct', 'median'),
    Tree_Density_Mean=('tree_density_per_km2', 'mean'),
    Tree_Density_Median=('tree_density_per_km2', 'median'),
    Canopy_Cover_Mean=('canopy_cover_pct_corrected', 'mean'),
    Canopy_Cover_Median=('canopy_cover_pct_corrected', 'median'),
    Crown_Area_Mean=('mean_crown_area_m2', 'mean'),
    Crown_Area_Median=('mean_crown_area_m2', 'median')
).reset_index()

income_q_path = os.path.join(OUT_DIR, "austin_income_quartile_summary.csv")
income_q_summary.to_csv(income_q_path, index=False)

# Q1 Subgroup Comparison (High vs Low within Q1)
q1_df = merged_df[merged_df['income_quartile'] == 'Q1 (Low)'].copy()
median_td_q1 = q1_df['tree_density_per_km2'].median()
median_ca_q1 = q1_df['mean_crown_area_m2'].median()
median_cc_q1 = q1_df['canopy_cover_pct_corrected'].median()

q1_subgroups = pd.DataFrame([
    {
        'Subgroup': 'Q1 All',
        'N': len(q1_df),
        'Dist_CBD_Mean': q1_df['dist_cbd_km'].mean(),
        'Dist_CBD_Median': q1_df['dist_cbd_km'].median(),
        'Housing_Density_Median': q1_df['housing_density_km2'].median(),
        'Park_Pct_Median': q1_df['park_pct'].median()
    },
    {
        'Subgroup': 'Q1 High Tree Density (>= Median)',
        'N': len(q1_df[q1_df['tree_density_per_km2'] >= median_td_q1]),
        'Dist_CBD_Mean': q1_df[q1_df['tree_density_per_km2'] >= median_td_q1]['dist_cbd_km'].mean(),
        'Dist_CBD_Median': q1_df[q1_df['tree_density_per_km2'] >= median_td_q1]['dist_cbd_km'].median(),
        'Housing_Density_Median': q1_df[q1_df['tree_density_per_km2'] >= median_td_q1]['housing_density_km2'].median(),
        'Park_Pct_Median': q1_df[q1_df['tree_density_per_km2'] >= median_td_q1]['park_pct'].median()
    },
    {
        'Subgroup': 'Q1 Low Tree Density (< Median)',
        'N': len(q1_df[q1_df['tree_density_per_km2'] < median_td_q1]),
        'Dist_CBD_Mean': q1_df[q1_df['tree_density_per_km2'] < median_td_q1]['dist_cbd_km'].mean(),
        'Dist_CBD_Median': q1_df[q1_df['tree_density_per_km2'] < median_td_q1]['dist_cbd_km'].median(),
        'Housing_Density_Median': q1_df[q1_df['tree_density_per_km2'] < median_td_q1]['housing_density_km2'].median(),
        'Park_Pct_Median': q1_df[q1_df['tree_density_per_km2'] < median_td_q1]['park_pct'].median()
    },
    {
        'Subgroup': 'Q1 High Crown Area (>= Median)',
        'N': len(q1_df[q1_df['mean_crown_area_m2'] >= median_ca_q1]),
        'Dist_CBD_Mean': q1_df[q1_df['mean_crown_area_m2'] >= median_ca_q1]['dist_cbd_km'].mean(),
        'Dist_CBD_Median': q1_df[q1_df['mean_crown_area_m2'] >= median_ca_q1]['dist_cbd_km'].median(),
        'Housing_Density_Median': q1_df[q1_df['mean_crown_area_m2'] >= median_ca_q1]['housing_density_km2'].median(),
        'Park_Pct_Median': q1_df[q1_df['mean_crown_area_m2'] >= median_ca_q1]['park_pct'].median()
    },
    {
        'Subgroup': 'Q1 Low Crown Area (< Median)',
        'N': len(q1_df[q1_df['mean_crown_area_m2'] < median_ca_q1]),
        'Dist_CBD_Mean': q1_df[q1_df['mean_crown_area_m2'] < median_ca_q1]['dist_cbd_km'].mean(),
        'Dist_CBD_Median': q1_df[q1_df['mean_crown_area_m2'] < median_ca_q1]['dist_cbd_km'].median(),
        'Housing_Density_Median': q1_df[q1_df['mean_crown_area_m2'] < median_ca_q1]['housing_density_km2'].median(),
        'Park_Pct_Median': q1_df[q1_df['mean_crown_area_m2'] < median_ca_q1]['park_pct'].median()
    },
    {
        'Subgroup': 'Q1 High Canopy Cover (>= Median)',
        'N': len(q1_df[q1_df['canopy_cover_pct_corrected'] >= median_cc_q1]),
        'Dist_CBD_Mean': q1_df[q1_df['canopy_cover_pct_corrected'] >= median_cc_q1]['dist_cbd_km'].mean(),
        'Dist_CBD_Median': q1_df[q1_df['canopy_cover_pct_corrected'] >= median_cc_q1]['dist_cbd_km'].median(),
        'Housing_Density_Median': q1_df[q1_df['canopy_cover_pct_corrected'] >= median_cc_q1]['housing_density_km2'].median(),
        'Park_Pct_Median': q1_df[q1_df['canopy_cover_pct_corrected'] >= median_cc_q1]['park_pct'].median()
    },
    {
        'Subgroup': 'Q1 Low Canopy Cover (< Median)',
        'N': len(q1_df[q1_df['canopy_cover_pct_corrected'] < median_cc_q1]),
        'Dist_CBD_Mean': q1_df[q1_df['canopy_cover_pct_corrected'] < median_cc_q1]['dist_cbd_km'].mean(),
        'Dist_CBD_Median': q1_df[q1_df['canopy_cover_pct_corrected'] < median_cc_q1]['dist_cbd_km'].median(),
        'Housing_Density_Median': q1_df[q1_df['canopy_cover_pct_corrected'] < median_cc_q1]['housing_density_km2'].median(),
        'Park_Pct_Median': q1_df[q1_df['canopy_cover_pct_corrected'] < median_cc_q1]['park_pct'].median()
    }
])
q1_subgroup_path = os.path.join(OUT_DIR, "austin_q1_subgroup_distance_analysis.csv")
q1_subgroups.to_csv(q1_subgroup_path, index=False)
print("Saved income quartile and Q1 subgroup tables.")

# -----------------------------------------------------------------------------
# 4. REGRESSION MODELS: UNADJUSTED, SEQUENTIAL ADJUSTMENT, ATTENUATION (TASK 3)
# -----------------------------------------------------------------------------
print("\nFitting sequential OLS models with HC3 robust standard errors...")
outcomes = [
    ('tree_density_log10', 'Tree Density (log10 trees/km²)', 'Tree Density'),
    ('canopy_cover_log10', 'Canopy Cover (log10 %)', 'Canopy Cover'),
    ('mean_crown_area_log10', 'Mean Crown Area (log10 m²)', 'Mean Crown Area')
]

model_specs = [
    ('Model 1: Unadjusted', ['income_zscore']),
    ('Model 2: + Distance to Downtown', ['income_zscore', 'dist_cbd_km']),
    ('Model 3: + Distance + Housing Density', ['income_zscore', 'dist_cbd_km', 'housing_density_log10']),
    ('Model 4: Full Model (+ Park Coverage)', ['income_zscore', 'dist_cbd_km', 'housing_density_log10', 'park_pct'])
]

reg_results = []
unadj_betas = {}

for y_col, y_label, short_name in outcomes:
    for m_label, preds in model_specs:
        sub = merged_df.dropna(subset=[y_col] + preds).copy()
        X = sm.add_constant(sub[preds])
        y = sub[y_col]
        mod = sm.OLS(y, X).fit(cov_type='HC3')
        ci = mod.conf_int()
        
        income_b = mod.params['income_zscore']
        income_se = mod.bse['income_zscore']
        income_p = mod.pvalues['income_zscore']
        ci_low = ci.loc['income_zscore', 0]
        ci_high = ci.loc['income_zscore', 1]
        
        if m_label == 'Model 1: Unadjusted':
            unadj_betas[short_name] = income_b
            attenuation_pct = 0.0
        else:
            base_b = unadj_betas[short_name]
            attenuation_pct = ((base_b - income_b) / base_b) * 100.0 if base_b != 0 else np.nan
            
        row = {
            'Outcome': short_name,
            'Outcome_Transform': y_label,
            'Model': m_label,
            'N': len(sub),
            'R_squared': mod.rsquared,
            'Adj_R_squared': mod.rsquared_adj,
            'Income_beta': income_b,
            'Income_SE_HC3': income_se,
            'Income_95CI_low': ci_low,
            'Income_95CI_high': ci_high,
            'Income_pvalue': income_p,
            'Income_attenuation_pct': attenuation_pct
        }
        
        # Add covariate coefficients
        for cov in ['dist_cbd_km', 'housing_density_log10', 'park_pct']:
            if cov in preds:
                row[f'{cov}_coef'] = mod.params[cov]
                row[f'{cov}_se_hc3'] = mod.bse[cov]
                row[f'{cov}_pvalue'] = mod.pvalues[cov]
            else:
                row[f'{cov}_coef'] = np.nan
                row[f'{cov}_se_hc3'] = np.nan
                row[f'{cov}_pvalue'] = np.nan
                
        reg_results.append(row)

reg_df = pd.DataFrame(reg_results)
reg_csv_path = os.path.join(OUT_DIR, "austin_multivariable_sequential_regressions.csv")
reg_df.to_csv(reg_csv_path, index=False)
print("Saved sequential regression results.")

# -----------------------------------------------------------------------------
# 5. SPATIAL ERROR MODELS (TASK 3)
# -----------------------------------------------------------------------------
print("\nFitting spatial error models (GM_Error_Het via PySAL spreg)...")
gdf_utm = merged_gdf.to_crs(epsg=32614)
centroids = np.column_stack([gdf_utm.geometry.centroid.x, gdf_utm.geometry.centroid.y])

spatial_results = []
for y_col, y_label, short_name in outcomes:
    sub = merged_gdf.dropna(subset=[y_col, 'income_zscore', 'dist_cbd_km', 'housing_density_log10', 'park_pct']).copy()
    sub_utm = sub.to_crs(epsg=32614)
    pts_sub = np.column_stack([sub_utm.geometry.centroid.x, sub_utm.geometry.centroid.y])
    
    w_knn = libpysal.weights.KNN.from_array(pts_sub, k=6)
    w_knn.transform = 'R'
    
    y = sub[y_col].values.reshape(-1, 1)
    
    # 1. Unadjusted Spatial Error
    x_unadj = sub[['income_zscore']].values
    mod_sp_unadj = GM_Error_Het(y, x_unadj, w=w_knn, name_y=y_col, name_x=['income_zscore'])
    
    # 2. Full Adjusted Spatial Error
    x_full = sub[['income_zscore', 'dist_cbd_km', 'housing_density_log10', 'park_pct']].values
    mod_sp_full = GM_Error_Het(y, x_full, w=w_knn, name_y=y_col, name_x=['income_zscore', 'dist_cbd_km', 'housing_density_log10', 'park_pct'])
    
    # Unadjusted stats
    b_unadj_sp = mod_sp_unadj.betas[1][0]
    se_unadj_sp = mod_sp_unadj.std_err[1]
    p_unadj_sp = mod_sp_unadj.z_stat[1][1]
    lam_unadj_sp = mod_sp_unadj.betas[-1][0]
    
    # Full stats
    b_full_sp = mod_sp_full.betas[1][0]
    se_full_sp = mod_sp_full.std_err[1]
    p_full_sp = mod_sp_full.z_stat[1][1]
    lam_full_sp = mod_sp_full.betas[-1][0]
    
    spatial_results.append({
        'Outcome': short_name,
        'N': len(sub),
        'Unadj_Income_beta_sp': b_unadj_sp,
        'Unadj_Income_SE_sp': se_unadj_sp,
        'Unadj_Income_pvalue_sp': p_unadj_sp,
        'Unadj_Spatial_Lambda': lam_unadj_sp,
        'Full_Income_beta_sp': b_full_sp,
        'Full_Income_SE_sp': se_full_sp,
        'Full_Income_pvalue_sp': p_full_sp,
        'Full_Spatial_Lambda': lam_full_sp,
        'Spatial_Income_Attenuation_pct': ((b_unadj_sp - b_full_sp) / b_unadj_sp) * 100.0 if b_unadj_sp != 0 else np.nan
    })

spatial_df = pd.DataFrame(spatial_results)
spatial_csv_path = os.path.join(OUT_DIR, "austin_spatial_error_models.csv")
spatial_df.to_csv(spatial_csv_path, index=False)
print("Saved spatial error model results.")

# -----------------------------------------------------------------------------
# 6. SENSITIVITY TO LARGE PERIPHERAL CENSUS BLOCK GROUPS (TASK 2, ITEM 5)
# -----------------------------------------------------------------------------
print("\nRunning sensitivity analyses for large peripheral block groups...")
sensitivity_cuts = [
    ('Full analytical sample', merged_df),
    ('Excl. area > 20 km²', merged_df[merged_df['block_area_km2'] <= 20.0]),
    ('Excl. area > 10 km²', merged_df[merged_df['block_area_km2'] <= 10.0]),
    ('Excl. top 5% largest BGs', merged_df[merged_df['block_area_km2'] <= merged_df['block_area_km2'].quantile(0.95)])
]

sens_rows = []
for spec_name, sub_sample in sensitivity_cuts:
    for y_col, y_label, short_name in outcomes:
        s = sub_sample.dropna(subset=[y_col, 'income_zscore', 'dist_cbd_km', 'housing_density_log10', 'park_pct'])
        
        # Unadjusted
        m1 = sm.OLS(s[y_col], sm.add_constant(s[['income_zscore']])).fit(cov_type='HC3')
        # Full adjusted
        m4 = sm.OLS(s[y_col], sm.add_constant(s[['income_zscore', 'dist_cbd_km', 'housing_density_log10', 'park_pct']])).fit(cov_type='HC3')
        
        ci1 = m1.conf_int()
        ci4 = m4.conf_int()
        
        sens_rows.append({
            'Specification': spec_name,
            'Outcome': short_name,
            'N': len(s),
            'Unadj_Income_beta': m1.params['income_zscore'],
            'Unadj_Income_SE': m1.bse['income_zscore'],
            'Unadj_Income_pvalue': m1.pvalues['income_zscore'],
            'Unadj_Income_CI_low': ci1.loc['income_zscore', 0],
            'Unadj_Income_CI_high': ci1.loc['income_zscore', 1],
            'Full_Income_beta': m4.params['income_zscore'],
            'Full_Income_SE': m4.bse['income_zscore'],
            'Full_Income_pvalue': m4.pvalues['income_zscore'],
            'Full_Income_CI_low': ci4.loc['income_zscore', 0],
            'Full_Income_CI_high': ci4.loc['income_zscore', 1],
            'Attenuation_pct': ((m1.params['income_zscore'] - m4.params['income_zscore']) / m1.params['income_zscore']) * 100.0
        })

sens_df = pd.DataFrame(sens_rows)
sens_csv_path = os.path.join(OUT_DIR, "austin_large_blockgroup_sensitivity.csv")
sens_df.to_csv(sens_csv_path, index=False)
print("Saved peripheral block group sensitivity results.")

# -----------------------------------------------------------------------------
# 7. GENERATE FIGURE A: AUSTIN SPATIAL PATTERNS (TASK 4)
# -----------------------------------------------------------------------------
print("\nGenerating Figure A: Austin Spatial Patterns (Maps)...")
fig, axes = plt.subplots(1, 3, figsize=(15.5, 6.2), dpi=300)
plt.subplots_adjust(wspace=0.15, left=0.04, right=0.96, top=0.92, bottom=0.08)

# Panel configs
map_configs = [
    {
        'col': 'median_income',
        'title': 'Median Household Income',
        'label': 'Income ($)',
        'cmap': 'viridis',
        'fmt': '${:,.0f}',
        'panel_letter': 'a'
    },
    {
        'col': 'mean_crown_area_m2',
        'title': 'Mean Individual Crown Area',
        'label': 'Mean Crown Area (m²)',
        'cmap': 'magma',
        'fmt': '{:.2f}',
        'panel_letter': 'b'
    },
    {
        'col': 'tree_density_per_km2',
        'title': 'Detected Tree Density',
        'label': 'Tree Density (trees/km²)',
        'cmap': 'YlGn',
        'fmt': '{:,.0f}',
        'panel_letter': 'c'
    }
]

# Convert to Web Mercator or UTM for plotting with scale bar, or use EPSG:32614
gdf_map = merged_gdf.to_crs(epsg=32614)
bnd_map = gdf_bnd.to_crs(epsg=32614)
cbd_point = gpd.GeoSeries([Point(CBD_LON, CBD_LAT)], crs='EPSG:4326').to_crs(epsg=32614).iloc[0]

# Compute common bounds
minx, miny, maxx, maxy = bnd_map.total_bounds
dx = maxx - minx
dy = maxy - miny
xlim = (minx - 0.03 * dx, maxx + 0.03 * dx)
ylim = (miny - 0.03 * dy, maxy + 0.03 * dy)

for ax, cfg in zip(axes, map_configs):
    # Plot background boundary
    bnd_map.plot(ax=ax, facecolor='#f7f7f7', edgecolor='#444444', linewidth=0.9, zorder=1)
    
    # Plot choropleth
    sub_map = gdf_map.dropna(subset=[cfg['col']])
    gdf_map.plot(
        column=cfg['col'],
        ax=ax,
        cmap=cfg['cmap'],
        linewidth=0.25,
        edgecolor='#ffffff',
        legend=True,
        legend_kwds={
            'orientation': 'horizontal',
            'pad': 0.04,
            'fraction': 0.046,
            'shrink': 0.75,
            'label': cfg['label']
        },
        missing_kwds={'color': '#dcdcdc', 'label': 'Missing'},
        zorder=2
    )
    
    # Plot Downtown CBD point
    ax.scatter(
        [cbd_point.x], [cbd_point.y],
        marker='*', s=220, facecolor='#d90429', edgecolor='#ffffff',
        linewidth=1.0, zorder=5, label='Downtown Austin (Capitol)'
    )
    
    # Format axes
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(cfg['title'], fontsize=12, fontweight='bold', pad=10)
    
    # Bold panel letter
    ax.text(
        0.03, 0.96, cfg['panel_letter'],
        transform=ax.transAxes,
        fontsize=16, fontweight='bold',
        va='top', ha='left',
        bbox=dict(boxstyle='square,pad=0.2', facecolor='white', edgecolor='none', alpha=0.85)
    )

# Add scale bar and north arrow to Panel a
ax0 = axes[0]
sb_x = minx + 0.08 * dx
sb_y = miny + 0.06 * dy
sb_len = 10000.0  # 10 km
ax0.plot([sb_x, sb_x + sb_len], [sb_y, sb_y], color='#111111', linewidth=2.5, zorder=6)
ax0.text(sb_x + sb_len / 2.0, sb_y - 0.02 * dy, '10 km', ha='center', va='top', fontsize=9, fontweight='bold', zorder=6)

# North arrow
na_x = maxx - 0.10 * dx
na_y = maxy - 0.08 * dy
ax0.annotate(
    'N', xy=(na_x, na_y), xytext=(na_x, na_y - 0.04 * dy),
    arrowprops=dict(facecolor='#111111', width=1.5, headwidth=6, headlength=7),
    ha='center', va='bottom', fontsize=10, fontweight='bold', zorder=6
)

# Add single reference marker legend in Panel a
ax0.legend(
    handles=[Line2D([0], [0], marker='*', color='w', markerfacecolor='#d90429', markersize=12, label='Downtown Austin (CBD)')],
    loc='lower left', frameon=True, framealpha=0.9, facecolor='white', edgecolor='none', fontsize=8.5
)

fig_a_png = os.path.join(OUT_DIR, "figure_a_austin_spatial_patterns.png")
fig_a_pdf = os.path.join(OUT_DIR, "figure_a_austin_spatial_patterns.pdf")
plt.savefig(fig_a_png, dpi=300, bbox_inches='tight')
plt.savefig(fig_a_pdf, bbox_inches='tight')
plt.close()
print("Saved Figure A (PNG and PDF).")

# -----------------------------------------------------------------------------
# 8. GENERATE FIGURE B: DISTANCE AND INCOME (TASK 4)
# -----------------------------------------------------------------------------
print("\nGenerating Figure B: Distance and Income Relationships...")
fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.5), dpi=300)
plt.subplots_adjust(hspace=0.28, wspace=0.26, left=0.08, right=0.95, top=0.94, bottom=0.08)

quartile_palette = {
    'Q1 (Low)': '#2b83ba',
    'Q2': '#abdda4',
    'Q3': '#fdae61',
    'Q4 (High)': '#d7191c'
}

# (a) Income vs Distance
ax = axes[0, 0]
sns.scatterplot(
    data=merged_df, x='dist_cbd_km', y='median_income',
    hue='income_quartile', palette=quartile_palette,
    alpha=0.65, s=32, edgecolor='none', ax=ax, zorder=2
)
sns.regplot(
    data=merged_df, x='dist_cbd_km', y='median_income',
    scatter=False, lowess=True, ax=ax,
    color='#111111', line_kws={'linewidth': 2.2, 'linestyle': '-', 'label': 'LOESS Trend', 'zorder': 3}
)
ax.set_xlabel('Distance to Downtown (km)', fontsize=10, fontweight='bold')
ax.set_ylabel('Median Household Income ($)', fontsize=10, fontweight='bold')
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
ax.set_title('Household Income vs. Distance', fontsize=11, fontweight='bold')
ax.text(0.04, 0.94, 'a', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
ax.grid(True, linestyle=':', alpha=0.5, zorder=1)
ax.legend(title='Income Quartile', frameon=True, framealpha=0.9, fontsize=8, title_fontsize=8.5, loc='upper left')

# (b) Mean Crown Area vs Distance
ax = axes[0, 1]
sns.scatterplot(
    data=merged_df, x='dist_cbd_km', y='mean_crown_area_m2',
    hue='income_quartile', palette=quartile_palette,
    alpha=0.65, s=32, edgecolor='none', ax=ax, zorder=2, legend=False
)
sns.regplot(
    data=merged_df.dropna(subset=['mean_crown_area_m2']), x='dist_cbd_km', y='mean_crown_area_m2',
    scatter=False, lowess=True, ax=ax,
    color='#111111', line_kws={'linewidth': 2.2, 'linestyle': '-', 'zorder': 3}
)
ax.set_xlabel('Distance to Downtown (km)', fontsize=10, fontweight='bold')
ax.set_ylabel('Mean Crown Area (m²)', fontsize=10, fontweight='bold')
ax.set_title('Mean Crown Area vs. Distance (ρ = -0.313, p < 10⁻¹⁴)', fontsize=11, fontweight='bold')
ax.text(0.04, 0.94, 'b', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
ax.grid(True, linestyle=':', alpha=0.5, zorder=1)

# (c) Tree Density vs Distance
ax = axes[1, 0]
sns.scatterplot(
    data=merged_df, x='dist_cbd_km', y='tree_density_per_km2',
    hue='income_quartile', palette=quartile_palette,
    alpha=0.65, s=32, edgecolor='none', ax=ax, zorder=2, legend=False
)
sns.regplot(
    data=merged_df, x='dist_cbd_km', y='tree_density_per_km2',
    scatter=False, lowess=True, ax=ax,
    color='#111111', line_kws={'linewidth': 2.2, 'linestyle': '-', 'zorder': 3}
)
ax.set_xlabel('Distance to Downtown (km)', fontsize=10, fontweight='bold')
ax.set_ylabel('Tree Density (trees/km²)', fontsize=10, fontweight='bold')
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
ax.set_title('Tree Density vs. Distance (ρ = +0.071, p = 0.078)', fontsize=11, fontweight='bold')
ax.text(0.04, 0.94, 'c', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
ax.grid(True, linestyle=':', alpha=0.5, zorder=1)

# (d) Canopy Cover vs Distance
ax = axes[1, 1]
sns.scatterplot(
    data=merged_df, x='dist_cbd_km', y='canopy_cover_pct_corrected',
    hue='income_quartile', palette=quartile_palette,
    alpha=0.65, s=32, edgecolor='none', ax=ax, zorder=2, legend=False
)
sns.regplot(
    data=merged_df, x='dist_cbd_km', y='canopy_cover_pct_corrected',
    scatter=False, lowess=True, ax=ax,
    color='#111111', line_kws={'linewidth': 2.2, 'linestyle': '-', 'zorder': 3}
)
ax.set_xlabel('Distance to Downtown (km)', fontsize=10, fontweight='bold')
ax.set_ylabel('Canopy Cover (%)', fontsize=10, fontweight='bold')
ax.set_title('Canopy Cover vs. Distance (ρ = -0.003, p = 0.948)', fontsize=11, fontweight='bold')
ax.text(0.04, 0.94, 'd', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
ax.grid(True, linestyle=':', alpha=0.5, zorder=1)

fig_b_png = os.path.join(OUT_DIR, "figure_b_distance_and_income.png")
fig_b_pdf = os.path.join(OUT_DIR, "figure_b_distance_and_income.pdf")
plt.savefig(fig_b_png, dpi=300, bbox_inches='tight')
plt.savefig(fig_b_pdf, bbox_inches='tight')
plt.close()
print("Saved Figure B (PNG and PDF).")

# -----------------------------------------------------------------------------
# 9. GENERATE FIGURE C: EXPLAINING INCOME-CANOPY RELATIONSHIP (FOREST PLOT) (TASK 4)
# -----------------------------------------------------------------------------
print("\nGenerating Figure C: Forest Plot of Income Coefficients Across Models...")
fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.2), dpi=300, sharey=True)
plt.subplots_adjust(wspace=0.18, left=0.22, right=0.96, top=0.88, bottom=0.12)

outcomes_for_c = [
    ('Tree Density', 'Tree Density (log10 trees/km²)', 'a'),
    ('Canopy Cover', 'Canopy Cover (log10 %)', 'b'),
    ('Mean Crown Area', 'Mean Crown Area (log10 m²)', 'c')
]

models_display = [
    ('Model 1: Unadjusted', 'Model 1: Unadjusted (Income only)', '#1f77b4'),
    ('Model 2: + Distance to Downtown', 'Model 2: + Distance to Downtown', '#ff7f0e'),
    ('Model 3: + Distance + Housing Density', 'Model 3: + Distance + Housing Density', '#2ca02c'),
    ('Model 4: Full Model (+ Park Coverage)', 'Model 4: Full Model (+ Park Coverage)', '#d62728')
]

model_keys = [m[0] for m in models_display]
model_labels = [m[1] for m in models_display]
model_colors = [m[2] for m in models_display]

y_pos = np.arange(len(models_display))[::-1]

for ax, (short_name, full_label, panel_letter) in zip(axes, outcomes_for_c):
    sub_df = reg_df[reg_df['Outcome'] == short_name].set_index('Model').reindex(model_keys)
    
    # Vertical line at null
    ax.axvline(0, color='#666666', linestyle='--', linewidth=1.0, zorder=1)
    
    # Plot point estimates and CIs
    for i, (m_key, m_lbl, col) in enumerate(zip(model_keys, model_labels, model_colors)):
        row = sub_df.loc[m_key]
        b = row['Income_beta']
        ci_l = row['Income_95CI_low']
        ci_h = row['Income_95CI_high']
        p_val = row['Income_pvalue']
        
        y_val = y_pos[i]
        
        # Error bar
        ax.plot([ci_l, ci_h], [y_val, y_val], color=col, linewidth=2.2)
        # Marker
        ax.plot(b, y_val, marker='s', markersize=8, color=col, markeredgecolor='black', markeredgewidth=0.8, zorder=4)
        
        # Annotation text
        sig_star = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ' (ns)'
        label_text = f"β = {b:.4f}{sig_star}\n95% CI [{ci_l:.4f}, {ci_h:.4f}]"
        
        # Position label above or beside marker
        ax.text(
            ci_h + 0.003 if b >= 0 else ci_l - 0.003,
            y_val,
            f"p={p_val:.3f}",
            va='center',
            ha='left' if b >= 0 else 'right',
            fontsize=7.5,
            color='#333333'
        )

    ax.set_yticks(y_pos)
    if ax == axes[0]:
        ax.set_yticklabels(model_labels, fontsize=9.5, fontweight='bold')
    else:
        ax.set_yticklabels([])
        
    ax.set_title(short_name, fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel('Income Standardized Coefficient (β)', fontsize=9.5, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.5, zorder=1)
    
    # Panel letter
    ax.text(0.04, 0.94, panel_letter, transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')

fig_c_png = os.path.join(OUT_DIR, "figure_c_explaining_income_canopy_relationship.png")
fig_c_pdf = os.path.join(OUT_DIR, "figure_c_explaining_income_canopy_relationship.pdf")
plt.savefig(fig_c_png, dpi=300, bbox_inches='tight')
plt.savefig(fig_c_pdf, bbox_inches='tight')
plt.close()
print("Saved Figure C (PNG and PDF).")

print("Analysis and figure generation completed successfully!")
