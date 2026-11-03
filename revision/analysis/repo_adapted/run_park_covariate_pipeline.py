#!/usr/bin/env python3
"""
run_park_covariate_pipeline.py
Calculates parkland and natural-area coverage for each census unit in the 4-city equity study.
Uses official municipal GIS datasets:
  - Austin: City of Austin parkland boundaries (dataset 8f2b-a4q5 / v8hw-gz65)
  - Bloomington: City of Bloomington Parks GIS polygon dataset (Layer 13)
  - Cupertino: City of Cupertino Parks polygon layer (PropertyInfo_Internal2/MapServer/22)
  - Surrey: City of Surrey OpenData MapServer (Parks layer 256 and Park Natural Areas layer 245)

Intersects non-overlapping dissolved parkland polygon unions with each census unit's
municipal-clipped geometry in local UTM projections.
Preserves the exact analytical sample of 1,283 census units.
Saves results to analysis_output/equity_results/all_cities_park_covariates.csv.
"""

import os
import urllib.request, ssl, json
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.validation import make_valid
from shapely import STRtree

ctx = ssl._create_unverified_context()

def fetch_geojson(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, context=ctx) as resp:
        gdf = gpd.read_file(resp)
        gdf['geometry'] = gdf['geometry'].apply(lambda g: make_valid(g) if g and not g.is_valid else g)
        return gdf

# Census units come from the committed covariate table and boundaries (staged by
# repro_data); park layers are downloaded live from each city's GIS service, so
# the result can drift if a city edits its park data. The committed output is
# data/covariates/all_cities_park_covariates.csv.
import sys
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
import repro_data  # noqa: E402

repro_data.stage_covariates()
for _city in repro_data.CITIES:
    repro_data.ensure_boundaries(_city)

BASE_DIR = repro_data.BASE_DIR
EQUITY_DIR = repro_data.EQUITY_STAGE_DIR
MASTER_CSV = os.path.join(EQUITY_DIR, "all_cities_urban_covariates_corrected.csv")

cov_df = pd.read_csv(MASTER_CSV)
print(f"Loaded master cohort: {len(cov_df)} census units across {cov_df['city'].nunique()} cities.")

utm_epsg = {
    'austin': 32614,
    'bloomington': 32616,
    'cupertino': 32610,
    'surrey': 32610
}

# 1. Fetch park GIS layers
print("\n--- 1. FETCHING OFFICIAL GIS PARK DATASETS ---")
print("Fetching Austin parkland (Socrata 8f2b-a4q5 / v8hw-gz65)...")
austin_parks = fetch_geojson('https://data.austintexas.gov/api/geospatial/v8hw-gz65?method=export&format=GeoJSON')
print(f"  Austin features: {len(austin_parks)}")

print("Fetching Bloomington parks (DataPortal Layer 13)...")
bl_parks = fetch_geojson('https://bloomington.in.gov/arcgis-server/rest/services/DataPortal/DataPortal_ParksAndRec/MapServer/13/query?where=1=1&outFields=*&f=geojson')
print(f"  Bloomington features: {len(bl_parks)}")

print("Fetching Cupertino parks (PropertyInfo_Internal2 Layer 22)...")
cup_parks = fetch_geojson('https://gis.cupertino.org/cupgis/rest/services/Public/PropertyInfo_Internal2/MapServer/22/query?where=1=1&outFields=*&f=geojson')
print(f"  Cupertino features: {len(cup_parks)}")

print("Fetching Surrey parks (Layer 256) & natural areas (Layer 245)...")
surrey_parks = fetch_geojson('https://gisservices.surrey.ca/arcgis/rest/services/OpenData/MapServer/256/query?where=1=1&outFields=*&f=geojson')
surrey_nat = fetch_geojson('https://gisservices.surrey.ca/arcgis/rest/services/OpenData/MapServer/245/query?where=1=1&outFields=*&f=geojson')
print(f"  Surrey parks: {len(surrey_parks)}, natural areas: {len(surrey_nat)}")

# 2. Build non-overlapping dissolved unions
print("\n--- 2. BUILDING DISSOLVED NON-OVERLAPPING UNIONS ---")
park_unions_total = {}
park_unions_natural = {}

# AUSTIN
epsg_atx = utm_epsg['austin']
atx_utm = austin_parks.to_crs(epsg=epsg_atx)
park_unions_total['austin'] = make_valid(atx_utm.geometry.union_all())
# Natural lands in Austin: tpl_landuse == 'Natural Lands' or park_type in ['Nature Preserve', 'Greenbelt']
atx_nat_sub = atx_utm[
    (atx_utm['tpl_landuse'] == 'Natural Lands') | 
    (atx_utm['park_type'].isin(['Nature Preserve', 'Greenbelt']))
]
park_unions_natural['austin'] = make_valid(atx_nat_sub.geometry.union_all())

# BLOOMINGTON
epsg_bl = utm_epsg['bloomington']
bl_utm = bl_parks.to_crs(epsg=epsg_bl)
park_unions_total['bloomington'] = make_valid(bl_utm.geometry.union_all())
# Nature Preserve in Bloomington
bl_nat_sub = bl_utm[bl_utm['type'] == 'Nature Preserve']
park_unions_natural['bloomington'] = make_valid(bl_nat_sub.geometry.union_all()) if len(bl_nat_sub) > 0 else None

# CUPERTINO
epsg_cup = utm_epsg['cupertino']
cup_utm = cup_parks.to_crs(epsg=epsg_cup)
park_unions_total['cupertino'] = make_valid(cup_utm.geometry.union_all())
# Environmental Area in Cupertino
cup_nat_sub = cup_utm[cup_utm['EnvironmentalArea'].astype(str).str.lower().isin(['yes', '1', 'true'])]
park_unions_natural['cupertino'] = make_valid(cup_nat_sub.geometry.union_all()) if len(cup_nat_sub) > 0 else None

# SURREY
epsg_sry = utm_epsg['surrey']
sry_p_utm = surrey_parks.to_crs(epsg=epsg_sry)
sry_na_utm = surrey_nat.to_crs(epsg=epsg_sry)
# Total parkland = Union of Parks and Natural Areas (non-overlapping)
sry_total_series = gpd.GeoSeries(list(sry_p_utm.geometry) + list(sry_na_utm.geometry), crs=f"EPSG:{epsg_sry}")
park_unions_total['surrey'] = make_valid(sry_total_series.union_all())
park_unions_natural['surrey'] = make_valid(sry_na_utm.geometry.union_all())

for city in ['austin', 'bloomington', 'cupertino', 'surrey']:
    t_a = park_unions_total[city].area / 1e6
    n_a = park_unions_natural[city].area / 1e6 if park_unions_natural[city] else 0.0
    print(f"  {city.upper()}: Total Parkland = {t_a:.2f} km² | Dedicated Natural Areas = {n_a:.2f} km²")

# 3. Intersect with exact municipal-clipped census block groups
print("\n--- 3. CALCULATING PARK COVERAGE PER CENSUS UNIT ---")
results = []

for city in ['austin', 'bloomington', 'cupertino', 'surrey']:
    bg_path = os.path.join(BASE_DIR, 'census_boundaries', 'exact_boundaries', f'{city}_exact_block_groups.geojson')
    bg_gdf = gpd.read_file(bg_path)
    epsg = utm_epsg[city]
    bg_utm = bg_gdf.to_crs(epsg=epsg)
    bg_utm['geometry'] = bg_utm['geometry'].apply(lambda g: make_valid(g) if g and not g.is_valid else g)
    
    id_col = 'DAUID' if city == 'surrey' else 'GEOID'
    if city == 'surrey':
        bg_utm['clean_id'] = bg_utm[id_col].astype(str).str.zfill(8)
    else:
        bg_utm['clean_id'] = bg_utm[id_col].astype(str).apply(lambda s: s.split('US')[-1] if 'US' in s else s).str.zfill(12)
        
    bg_geom_dict = dict(zip(bg_utm['clean_id'], bg_utm.geometry))
    
    city_cov = cov_df[cov_df['city'] == city].copy()
    city_cov['clean_id_str'] = city_cov['clean_id'].astype(str)
    if city == 'surrey':
        city_cov['clean_id_str'] = city_cov['clean_id_str'].str.zfill(8)
    else:
        city_cov['clean_id_str'] = city_cov['clean_id_str'].str.zfill(12)
        
    u_total = park_unions_total[city]
    u_nat = park_unions_natural[city]
    
    for row in city_cov.itertuples():
        cid = getattr(row, 'clean_id_str')
        geoid = getattr(row, 'GEOID')
        
        unit_geom = bg_geom_dict.get(cid)
        if unit_geom is None or unit_geom.is_empty:
            print(f"Warning: geometry missing for {city} unit {cid}")
            continue
            
        clipped_area_m2 = unit_geom.area
        
        # Total park intersection
        if unit_geom.intersects(u_total):
            p_inter = unit_geom.intersection(u_total)
            park_area_m2 = p_inter.area
        else:
            park_area_m2 = 0.0
            
        # Natural area intersection
        if u_nat and unit_geom.intersects(u_nat):
            n_inter = unit_geom.intersection(u_nat)
            natural_area_m2 = n_inter.area
        else:
            natural_area_m2 = 0.0
            
        park_pct = min(100.0, (park_area_m2 / clipped_area_m2) * 100.0) if clipped_area_m2 > 0 else 0.0
        natural_pct = min(100.0, (natural_area_m2 / clipped_area_m2) * 100.0) if clipped_area_m2 > 0 else 0.0
        
        results.append({
            'GEOID': geoid,
            'clean_id': cid,
            'city': city,
            'unit_clipped_area_m2': clipped_area_m2,
            'unit_clipped_area_km2': clipped_area_m2 / 1e6,
            'park_area_m2': park_area_m2,
            'park_area_km2': park_area_m2 / 1e6,
            'park_pct': park_pct,
            'natural_area_m2': natural_area_m2,
            'natural_area_km2': natural_area_m2 / 1e6,
            'natural_area_pct': natural_pct,
            'developed_park_pct': max(0.0, park_pct - natural_pct)
        })

park_df = pd.DataFrame(results)
print(f"Computed park coverage for {len(park_df)} / {len(cov_df)} cohort units.")
assert len(park_df) == len(cov_df), f"Mismatch in sample size: {len(park_df)} vs {len(cov_df)}"

# 4. Save separate park coverage dataset
out_csv = os.path.join(EQUITY_DIR, "all_cities_park_covariates.csv")
park_df.to_csv(out_csv, index=False)
print(f"\nSaved park covariate dataset: {out_csv}")

# 5. Coverage audit summary
print("\n" + "="*80)
print("PARK COVERAGE AUDIT SUMMARY")
print("="*80)

audit_rows = []
for city in ['austin', 'bloomington', 'cupertino', 'surrey', 'ALL']:
    if city == 'ALL':
        c_sub = park_df
    else:
        c_sub = park_df[park_df['city'] == city]
        
    n_units = len(c_sub)
    n_with_park = (c_sub['park_pct'] > 0).sum()
    pct_units_with_park = (n_with_park / n_units) * 100.0
    
    audit_rows.append({
        'city': city.title(),
        'N_units': n_units,
        'N_with_park': n_with_park,
        'pct_units_with_park': f"{pct_units_with_park:.1f}%",
        'park_pct_mean': c_sub['park_pct'].mean(),
        'park_pct_median': c_sub['park_pct'].median(),
        'park_pct_q75': c_sub['park_pct'].quantile(0.75),
        'park_pct_max': c_sub['park_pct'].max(),
        'natural_pct_mean': c_sub['natural_area_pct'].mean(),
        'natural_pct_max': c_sub['natural_area_pct'].max()
    })

audit_df = pd.DataFrame(audit_rows)
print(audit_df.to_string(index=False))

# Also save audit summary
audit_out_csv = os.path.join(EQUITY_DIR, "park_coverage_audit_summary.csv")
audit_df.to_csv(audit_out_csv, index=False)
print(f"\nSaved audit summary table: {audit_out_csv}")
