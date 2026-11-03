#!/usr/bin/env python3
"""
analyze_independent_city_characteristics.py
-------------------------------------------
Quantifies independent climatic, urban-form, and environmental characteristics
for Austin, Bloomington, Cupertino, and Surrey to address Reviewer 2, Comment 4.

Uses strictly external, independent datasets:
  - Beck et al. (2018) Köppen-Geiger Climate Classification (1 km)
  - JRC GHSL GHS-BUILT-S R2023A Built-Up Surface Grid (30 arc-sec / 1 km)
  - Natural Earth Admin-1 Boundaries (CONUS and Canada)
  - U.S. Census Bureau 2020 Decennial Census
  - Statistics Canada 2021 Census of Population
  - NOAA NCEI 1991-2020 Climate Normals
  - Environment and Climate Change Canada Climate Normals
  - Commission for Environmental Cooperation (CEC) Ecological Regions of North America
"""

import os
import rasterio
from rasterio.features import rasterize
from rasterio.windows import from_bounds
from rasterio.mask import mask
import geopandas as gpd
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "independent_city_justification", "data")
OUT_DIR = os.path.join(BASE_DIR, "independent_city_justification")
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    print("=" * 80)
    print("INDEPENDENT CITY SELECTION ANALYSIS (REVIEWER 2, COMMENT 4)")
    print("=" * 80)

    # 1. Load boundaries
    admin1_path = os.path.join(DATA_DIR, "ne_50m_admin_1_states_provinces.geojson")
    admin1 = gpd.read_file(admin1_path)
    conus_gdf = admin1[(admin1['admin'] == "United States of America") & (~admin1['name'].isin(["Alaska", "Hawaii"]))]
    canada_gdf = admin1[admin1['admin'] == "Canada"]

    conus_geom = [conus_gdf.union_all()]
    canada_geom = [canada_gdf.union_all()]

    # 2. Climate & GHSL rasters
    kg_path = os.path.join(DATA_DIR, "Beck_KG_V1_present_0p0083.tif")
    ghs_path = os.path.join(DATA_DIR, "GHS_BUILT_S_E2020_GLOBE_R2023A_4326_30ss_V1_0.tif")
    na_bounds = (-142.0, 24.0, -52.0, 84.0)

    # Legend
    legend = {}
    with open(os.path.join(DATA_DIR, "legend.txt")) as f:
        for line in f:
            line = line.strip()
            if ":" in line and not line.startswith("Legend") and not line.startswith("The"):
                parts = line.split(":")
                try:
                    c_code = int(parts[0].strip())
                    rest = parts[1].strip().split()
                    c_sym = rest[0]
                    c_desc = " ".join(rest[1:]).split("[")[0].strip()
                    legend[c_code] = (c_sym, c_desc)
                except Exception:
                    pass

    with rasterio.open(ghs_path) as ghs_src, rasterio.open(kg_path) as kg_src:
        win_ghs = from_bounds(*na_bounds, transform=ghs_src.transform).round_offsets().round_lengths()
        transform_win = rasterio.windows.transform(win_ghs, ghs_src.transform)
        ghs_arr = ghs_src.read(1, window=win_ghs)
        win_kg = from_bounds(*na_bounds, transform=kg_src.transform).round_offsets().round_lengths()
        kg_arr = kg_src.read(1, window=win_kg)

        min_h = min(ghs_arr.shape[0], kg_arr.shape[0])
        min_w = min(ghs_arr.shape[1], kg_arr.shape[1])
        ghs_arr = ghs_arr[:min_h, :min_w]
        kg_arr = kg_arr[:min_h, :min_w]

        conus_mask = rasterize(conus_geom, out_shape=(min_h, min_w), transform=transform_win, fill=0, default_value=1, dtype=np.uint8) == 1
        canada_mask = rasterize(canada_geom, out_shape=(min_h, min_w), transform=transform_win, fill=0, default_value=1, dtype=np.uint8) == 1
        comb_mask = conus_mask | canada_mask

        tot_conus = ghs_arr[conus_mask].astype(np.float64).sum() / 1e6
        tot_can = ghs_arr[canada_mask].astype(np.float64).sum() / 1e6
        tot_comb = ghs_arr[comb_mask].astype(np.float64).sum() / 1e6

        coverage_rows = []
        for code in sorted(legend.keys()):
            sym, desc = legend[code]
            b_conus = ghs_arr[conus_mask & (kg_arr == code)].astype(np.float64).sum() / 1e6
            b_can = ghs_arr[canada_mask & (kg_arr == code)].astype(np.float64).sum() / 1e6
            b_comb = ghs_arr[comb_mask & (kg_arr == code)].astype(np.float64).sum() / 1e6

            if b_comb > 0.1:
                coverage_rows.append({
                    'kg_code': code,
                    'climate_class': sym,
                    'description': desc,
                    'is_study_climate': "YES" if code in [9, 14, 15, 25] else "No",
                    'representative_city': {9: "Cupertino", 14: "Austin", 15: "Surrey", 25: "Bloomington"}.get(code, "-"),
                    'conus_built_km2': b_conus,
                    'conus_pct': (b_conus / tot_conus) * 100.0,
                    'canada_built_km2': b_can,
                    'canada_pct': (b_can / tot_can) * 100.0,
                    'combined_built_km2': b_comb,
                    'combined_pct': (b_comb / tot_comb) * 100.0
                })

    cov_df = pd.DataFrame(coverage_rows).sort_values(by="combined_built_km2", ascending=False)
    cov_out = os.path.join(OUT_DIR, "climate_zone_built_up_coverage.csv")
    cov_df.to_csv(cov_out, index=False)
    print(f"[1/2] Saved climate zone built-up coverage table: {cov_out}")

    # Print summary
    study_sub = cov_df[cov_df['is_study_climate'] == "YES"]
    print("Study Climate Classes Built-Up Area Summary:")
    print(study_sub[['climate_class', 'representative_city', 'conus_pct', 'canada_pct', 'combined_pct']].to_string(index=False))
    print(f"CONUS 4-Class Total:    {study_sub['conus_pct'].sum():.2f}% ({study_sub['conus_built_km2'].sum():,.1f} km2)")
    print(f"Canada 4-Class Total:   {study_sub['canada_pct'].sum():.2f}% ({study_sub['canada_built_km2'].sum():,.1f} km2)")
    print(f"Combined 4-Class Total: {study_sub['combined_pct'].sum():.2f}% ({study_sub['combined_built_km2'].sum():,.1f} km2)")

    # 3. Municipal Built-Up Surface Calculation
    ghs_fractions = {}
    with rasterio.open(ghs_path) as src:
        for c in ['austin', 'bloomington', 'cupertino', 'surrey']:
            bp = os.path.join(BASE_DIR, "census_boundaries", "exact_boundaries", f"{c}_exact_boundary.geojson")
            bgdf = gpd.read_file(bp).to_crs("EPSG:4326")
            geom = [bgdf.union_all()]
            out_img, _ = mask(src, geom, crop=True)
            b_m2 = out_img[0].astype(np.float64).sum()
            b_km2 = b_m2 / 1e6
            
            utm_epsg = {'austin': 32614, 'bloomington': 32616, 'cupertino': 32610, 'surrey': 32610}[c]
            muni_km2 = bgdf.to_crs(epsg=utm_epsg).geometry.area.sum() / 1e6
            ghs_fractions[c] = (b_km2 / muni_km2) * 100.0

    # 4. Independent City-Level Comparison Table
    city_comparison = [
        {
            'City': "Austin, TX",
            'Country': "United States",
            'Koppen_Geiger_Class': "Cfa (Humid subtropical)",
            'CEC_Level_I_Ecoregion': "Great Plains (9)",
            'CEC_Level_III_Ecoregion': "Texas Blackland Prairies (9.4.7) / Edwards Plateau (9.4.6)",
            'Mean_Annual_Temp_C': 20.8,
            'Mean_Annual_Temp_F': 69.4,
            'Mean_Annual_Precip_mm': 920.8,
            'Mean_Annual_Precip_in': 36.25,
            'Climate_Data_Source': "NOAA NCEI 1991-2020 Normals (Austin Camp Mabry)",
            'Municipal_Land_Area_km2': 828.6,
            'Municipal_Land_Area_sqmi': 319.94,
            'Population_Decennial': 961855,
            'Population_Year': 2020,
            'Population_Density_km2': 1160.8,
            'Housing_Units_Decennial': 444437,
            'Housing_Density_km2': 536.4,
            'Demographic_Source': "U.S. Census Bureau 2020 Decennial Census (Tables P1, H1)",
            'Municipal_Parkland_Coverage_pct': 10.5,
            'Parkland_Source': "City of Austin GIS (v8hw-gz65) / TPL ParkScore 2023",
            'GHSL_Built_Up_Surface_pct': round(ghs_fractions['austin'], 1)
        },
        {
            'City': "Bloomington, IN",
            'Country': "United States",
            'Koppen_Geiger_Class': "Dfa (Humid continental, hot summer)",
            'CEC_Level_I_Ecoregion': "Eastern Temperate Forests (8)",
            'CEC_Level_III_Ecoregion': "Interior Plateau (8.3.3)",
            'Mean_Annual_Temp_C': 12.1,
            'Mean_Annual_Temp_F': 53.8,
            'Mean_Annual_Precip_mm': 1191.8,
            'Mean_Annual_Precip_in': 46.92,
            'Climate_Data_Source': "NOAA NCEI 1991-2020 Normals (Bloomington BMG)",
            'Municipal_Land_Area_km2': 60.17,
            'Municipal_Land_Area_sqmi': 23.23,
            'Population_Decennial': 79168,
            'Population_Year': 2020,
            'Population_Density_km2': 1315.7,
            'Housing_Units_Decennial': 36454,
            'Housing_Density_km2': 605.8,
            'Demographic_Source': "U.S. Census Bureau 2020 Decennial Census (Tables P1, H1)",
            'Municipal_Parkland_Coverage_pct': 10.8,
            'Parkland_Source': "City of Bloomington Parks GIS (Layer 13)",
            'GHSL_Built_Up_Surface_pct': round(ghs_fractions['bloomington'], 1)
        },
        {
            'City': "Cupertino, CA",
            'Country': "United States",
            'Koppen_Geiger_Class': "Csb (Warm-summer Mediterranean)",
            'CEC_Level_I_Ecoregion': "Mediterranean California (11)",
            'CEC_Level_III_Ecoregion': "Central California Foothills and Coastal Mountains (11.1.1)",
            'Mean_Annual_Temp_C': 16.0,
            'Mean_Annual_Temp_F': 60.8,
            'Mean_Annual_Precip_mm': 397.5,
            'Mean_Annual_Precip_in': 15.65,
            'Climate_Data_Source': "NOAA NCEI 1991-2020 Normals (San Jose / Santa Clara Valley)",
            'Municipal_Land_Area_km2': 29.34,
            'Municipal_Land_Area_sqmi': 11.33,
            'Population_Decennial': 60381,
            'Population_Year': 2020,
            'Population_Density_km2': 2058.0,
            'Housing_Units_Decennial': 21987,
            'Housing_Density_km2': 749.4,
            'Demographic_Source': "U.S. Census Bureau 2020 Decennial Census (Tables P1, H1)",
            'Municipal_Parkland_Coverage_pct': 3.2,
            'Parkland_Source': "City of Cupertino GIS (PropertyInfo Layer 22)",
            'GHSL_Built_Up_Surface_pct': round(ghs_fractions['cupertino'], 1)
        },
        {
            'City': "Surrey, BC",
            'Country': "Canada",
            'Koppen_Geiger_Class': "Cfb (Temperate oceanic)",
            'CEC_Level_I_Ecoregion': "Marine West Coast Forests (7)",
            'CEC_Level_III_Ecoregion': "Puget Lowland / Georgia Depression (7.1.7)",
            'Mean_Annual_Temp_C': 10.4,
            'Mean_Annual_Temp_F': 50.7,
            'Mean_Annual_Precip_mm': 1562.0,
            'Mean_Annual_Precip_in': 61.50,
            'Climate_Data_Source': "Environment and Climate Change Canada Normals (Surrey Kwantlen Park)",
            'Municipal_Land_Area_km2': 316.11,
            'Municipal_Land_Area_sqmi': 122.05,
            'Population_Decennial': 568322,
            'Population_Year': 2021,
            'Population_Density_km2': 1797.9,
            'Housing_Units_Decennial': 195098,
            'Housing_Density_km2': 617.2,
            'Demographic_Source': "Statistics Canada 2021 Census of Population (Catalogue 98-316-X2021001)",
            'Municipal_Parkland_Coverage_pct': 10.1,
            'Parkland_Source': "City of Surrey OpenData (Layers 256 + 245)",
            'GHSL_Built_Up_Surface_pct': round(ghs_fractions['surrey'], 1)
        }
    ]

    city_df = pd.DataFrame(city_comparison)
    comp_out = os.path.join(OUT_DIR, "city_selection_independent_comparison.csv")
    city_df.to_csv(comp_out, index=False)
    print(f"[2/2] Saved independent city comparison table: {comp_out}")
    print("Independent City Comparison Table:")
    print(city_df[['City', 'Koppen_Geiger_Class', 'CEC_Level_I_Ecoregion', 'Mean_Annual_Temp_C', 'Mean_Annual_Precip_mm', 'Population_Density_km2', 'Housing_Density_km2', 'Municipal_Parkland_Coverage_pct', 'GHSL_Built_Up_Surface_pct']].to_string(index=False))

if __name__ == "__main__":
    main()
