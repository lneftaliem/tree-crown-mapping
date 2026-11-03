#!/usr/bin/env python3
"""
run_crown_area_correction_pipeline.py

Latitude correction of crown areas and census-unit re-aggregation for Austin,
Bloomington, Cupertino, and Surrey.

The original per-tree datasets store crown size as a pixel count
(``crown_area_px``). Image pixels are fixed in degrees, so their ground area
shrinks with latitude. This script converts each tree's pixel count to square
metres using the true pixel area at that tree's latitude:

    pixel_area_m2 = DEG_PER_PX[city]^2 * 111320^2 * cos(latitude)
    crown_area_m2 = crown_area_px * pixel_area_m2

and then re-aggregates tree count, crown area, and canopy cover to each census
block group (US) or dissemination area (Canada).

Inputs (all committed to this repository, staged automatically by repro_data):
    data/tree_points/<city>_ripley_points_with_income.geojson.gz[.partNN]
    data/covariates/all_cities_urban_covariates.csv

Outputs (written to $TREE_MAPPING_BASE_DIR/analysis_output/):
    equity_results/all_cities_urban_covariates_corrected.csv
    equity_results/crown_area_correction_verification.csv
    ripley_data_corrected/<city>_ripley_points_with_income_corrected.geojson
        (only with --write-points; these files are large, ~2.2 GB in total)

The corrected scatter figures are drawn by generate_plots_corrected.py.

Usage:
    python run_crown_area_correction_pipeline.py
    python run_crown_area_correction_pipeline.py --write-points
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
from scipy import stats

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
import repro_data  # noqa: E402

np.random.seed(42)

BASE_DIR = repro_data.BASE_DIR
EQUITY_OUT_DIR = repro_data.EQUITY_STAGE_DIR
OUTPUT_GEOJSON_DIR = repro_data.CORRECTED_STAGE_DIR

CITIES = ['austin', 'bloomington', 'cupertino', 'surrey']

# Only the columns needed for aggregation are read unless --write-points is set,
# which keeps memory use manageable for Austin (~1.1 GB GeoJSON).
AGG_COLUMNS = ['block_group_id', 'latitude', 'crown_area_px']


def load_city(city, write_points):
    import pyogrio

    src = repro_data.ensure_city_points(city)
    if src is None:
        raise FileNotFoundError(f"Committed point data for {city} not found")

    if write_points:
        gdf = pyogrio.read_dataframe(src)
        assert 'crown_area_px' in gdf.columns, "crown_area_px column missing!"
        repro_data.apply_crown_area_correction(gdf, city)
        os.makedirs(OUTPUT_GEOJSON_DIR, exist_ok=True)
        out_file = os.path.join(OUTPUT_GEOJSON_DIR,
                                repro_data.CORRECTED_POINTS_STEM.format(city=city))
        print(f"  Writing corrected dataset: {out_file}")
        pyogrio.write_dataframe(gdf, out_file, driver='GeoJSON')
        return gdf

    df = pyogrio.read_dataframe(src, columns=AGG_COLUMNS, read_geometry=False)
    return repro_data.apply_crown_area_correction(df, city)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument('--write-points', action='store_true',
                    help='also write the full corrected per-tree GeoJSONs')
    ap.add_argument('--cities', default=','.join(CITIES),
                    help='comma-separated subset of cities (default: all four)')
    args = ap.parse_args()
    cities = [c.strip().lower() for c in args.cities.split(',') if c.strip()]

    print("=" * 80)
    print("CROWN AREA CORRECTION AND CENSUS-UNIT AGGREGATION")
    print("=" * 80)

    repro_data.stage_covariates()
    os.makedirs(EQUITY_OUT_DIR, exist_ok=True)

    cov_path = os.path.join(EQUITY_OUT_DIR, "all_cities_urban_covariates.csv")
    cov_df = pd.read_csv(cov_path)
    print(f"Loaded master covariate dataset: {len(cov_df)} records")

    # City reference income statistics (for the within-city income z-score)
    city_ref_stats = {}
    for city in CITIES:
        c_sub = cov_df[cov_df['city'] == city]
        city_ref_stats[city] = {
            'mean': c_sub['median_income'].mean(),
            'std':  c_sub['median_income'].std()
        }
        print(f"  {city.title()} reference: N={len(c_sub)}, "
              f"mean=${city_ref_stats[city]['mean']:,.2f}, sd=${city_ref_stats[city]['std']:,.2f}")

    updated_census_rows = []
    verification_stats = []

    for city in cities:
        print(f"\nProcessing {city.upper()}...")
        gdf = load_city(city, args.write_points)

        n_trees_orig = len(gdf)
        print(f"  Total trees: {n_trees_orig:,}")
        print(f"  crown_area_m2: mean={gdf['crown_area_m2'].mean():.2f}, "
              f"median={gdf['crown_area_m2'].median():.2f} "
              f"(min={gdf['crown_area_m2'].min():.2f}, max={gdf['crown_area_m2'].max():.2f})")
        print(f"  crown_area_px: mean={gdf['crown_area_px'].mean():.2f}, "
              f"median={gdf['crown_area_px'].median():.2f}")

        # ── Census-unit aggregation ─────────────────────────────────────────
        city_cov = cov_df[cov_df['city'] == city].copy()
        ref_geoids = set(city_cov['GEOID'].astype(str))
        gdf_ref = gdf[gdf['block_group_id'].astype(str).isin(ref_geoids)]

        agg_df = gdf_ref.groupby('block_group_id').agg(
            tree_count=('crown_area_m2', 'count'),
            total_crown_area_m2=('crown_area_m2', 'sum'),
            mean_crown_area_m2=('crown_area_m2', 'mean'),
            median_crown_area_m2=('crown_area_m2', 'median'),
            total_crown_area_px=('crown_area_px', 'sum'),
            mean_crown_area_px=('crown_area_px', 'mean')
        ).reset_index()
        agg_df = agg_df.rename(columns={'block_group_id': 'merge_geoid'})
        agg_df['merge_geoid'] = agg_df['merge_geoid'].astype(str)

        cols_to_drop = ['tree_count', 'tree_area_m2', 'mean_crown_area_m2', 'canopy_cover_pct',
                        'income_zscore', 'total_crown_area_m2', 'total_crown_area_px',
                        'mean_crown_area_px', 'mean_crown_area_log10']
        city_cov_clean = city_cov.drop(columns=[c for c in cols_to_drop if c in city_cov.columns]).copy()
        city_cov_clean['merge_geoid'] = city_cov_clean['GEOID'].astype(str)

        merged = pd.merge(city_cov_clean, agg_df, on='merge_geoid', how='left')

        # Units with no detections get zero trees
        merged['tree_count'] = merged['tree_count'].fillna(0).astype(int)
        merged['total_crown_area_m2'] = merged['total_crown_area_m2'].fillna(0.0)
        merged['total_crown_area_px'] = merged['total_crown_area_px'].fillna(0.0)

        # Units with fewer than 10 trees keep a missing mean crown area
        merged.loc[merged['tree_count'] < 10, 'mean_crown_area_m2'] = np.nan
        merged.loc[merged['tree_count'] < 10, 'median_crown_area_m2'] = np.nan
        merged.loc[merged['tree_count'] < 10, 'mean_crown_area_px'] = np.nan
        merged['mean_crown_area_log10'] = np.where(merged['mean_crown_area_m2'].notna(),
                                                   np.log10(merged['mean_crown_area_m2'] + 1.0), np.nan)

        # Physical canopy cover (%)
        land_area_m2 = merged['block_area_km2'] * 1e6
        merged['canopy_cover_pct_corrected'] = np.where(
            land_area_m2 > 0, (merged['total_crown_area_m2'] / land_area_m2) * 100.0, 0.0)
        merged['canopy_cover_pct_uncorrected'] = np.where(
            land_area_m2 > 0, (merged['total_crown_area_px'] / land_area_m2) * 100.0, 0.0)
        merged['canopy_cover_log10'] = np.log10(merged['canopy_cover_pct_corrected'] + 1.0)

        # Within-city income z-score
        mu = city_ref_stats[city]['mean']
        sd = city_ref_stats[city]['std']
        merged['income_zscore'] = (merged['median_income'] - mu) / sd

        merged = merged.drop(columns=['merge_geoid'])
        updated_census_rows.append(merged)

        # ── Verification statistics ─────────────────────────────────────────
        crown_valid = merged[merged['tree_count'] >= 10].dropna(subset=['mean_crown_area_m2', 'income_zscore'])
        rho_ca_corr, p_ca_corr = stats.spearmanr(crown_valid['income_zscore'], crown_valid['mean_crown_area_m2'])
        rho_ca_orig, p_ca_orig = stats.spearmanr(crown_valid['income_zscore'], crown_valid['mean_crown_area_px'])
        rho_cc_corr, p_cc_corr = stats.spearmanr(merged['income_zscore'], merged['canopy_cover_pct_corrected'])
        rho_cc_orig, p_cc_orig = stats.spearmanr(merged['income_zscore'], merged['canopy_cover_pct_uncorrected'])
        rho_td, p_td = stats.spearmanr(merged['income_zscore'], merged['tree_density_per_km2'])

        verification_stats.append({
            'city': city,
            'total_trees': n_trees_orig,
            'n_units_total': len(merged),
            'n_units_crown': len(crown_valid),
            'orig_mean_crown_px': gdf['crown_area_px'].mean(),
            'orig_median_crown_px': gdf['crown_area_px'].median(),
            'corr_mean_crown_m2': gdf['crown_area_m2'].mean(),
            'corr_median_crown_m2': gdf['crown_area_m2'].median(),
            'unit_mean_crown_m2': crown_valid['mean_crown_area_m2'].mean(),
            'unit_median_crown_m2': crown_valid['mean_crown_area_m2'].median(),
            'orig_unit_canopy_mean': merged['canopy_cover_pct_uncorrected'].mean(),
            'orig_unit_canopy_median': merged['canopy_cover_pct_uncorrected'].median(),
            'corr_unit_canopy_mean': merged['canopy_cover_pct_corrected'].mean(),
            'corr_unit_canopy_median': merged['canopy_cover_pct_corrected'].median(),
            'rho_crown_orig': rho_ca_orig,
            'p_crown_orig': p_ca_orig,
            'rho_crown_corr': rho_ca_corr,
            'p_crown_corr': p_ca_corr,
            'rho_canopy_orig': rho_cc_orig,
            'p_canopy_orig': p_cc_orig,
            'rho_canopy_corr': rho_cc_corr,
            'p_canopy_corr': p_cc_corr,
            'rho_density': rho_td,
            'p_density': p_td
        })
        del gdf, gdf_ref

    master_updated_df = pd.concat(updated_census_rows, ignore_index=True)
    suffix = '' if cities == CITIES else '_' + '_'.join(cities)
    master_out_path = os.path.join(EQUITY_OUT_DIR, f"all_cities_urban_covariates_corrected{suffix}.csv")
    master_updated_df.to_csv(master_out_path, index=False)
    print(f"\nSaved updated master covariate dataset: {master_out_path}")

    v_out_path = os.path.join(EQUITY_OUT_DIR, f"crown_area_correction_verification{suffix}.csv")
    pd.DataFrame(verification_stats).to_csv(v_out_path, index=False)
    print(f"Saved verification table: {v_out_path}")


if __name__ == '__main__':
    main()
