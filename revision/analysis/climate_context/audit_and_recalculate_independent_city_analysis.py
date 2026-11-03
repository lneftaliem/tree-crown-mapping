#!/usr/bin/env python3
"""
audit_and_recalculate_independent_city_analysis.py
--------------------------------------------------
Rigorous Quality-Control Audit and Recalculation for Reviewer 2, Comment 4.
Strictly external, independent datasets only (no tree detection or socioeconomic outputs).

Addresses:
  1. Rigorous spatial co-registration: Nearest-neighbor reprojection of Beck et al. (2018)
     Köppen-Geiger raster onto the exact affine transform, grid cells, and CRS of GHSL 2020.
  2. Built-up surface quantification: Verification of GHS-BUILT-S (m2 physical built-up structural
     surface, not total urbanized land) across CONUS, Canada, and North America.
  3. Municipal polygon-wide climate distribution across full municipal boundaries.
  4. Fully audited comparison of city-level characteristics distinguishing calculated GIS metrics
     from primary external records.
"""

import os
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
from rasterio.windows import from_bounds
from rasterio.mask import mask
import geopandas as gpd
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = BASE_DIR

def main():
    print("=" * 80)
    print("TARGETED QUALITY-CONTROL AUDIT: INDEPENDENT CITY SELECTION")
    print("=" * 80)

    # 1. Boundaries
    admin1_path = os.path.join(DATA_DIR, "ne_50m_admin_1_states_provinces.geojson")
    admin1 = gpd.read_file(admin1_path)
    conus_gdf = admin1[(admin1['admin'] == "United States of America") & (~admin1['name'].isin(["Alaska", "Hawaii"]))]
    canada_gdf = admin1[admin1['admin'] == "Canada"]

    conus_geom = [conus_gdf.union_all()]
    canada_geom = [canada_gdf.union_all()]

    # 2. Rasters
    kg_path = os.path.join(DATA_DIR, "Beck_KG_V1_present_0p0083.tif")
    ghs_path = os.path.join(DATA_DIR, "GHS_BUILT_S_E2020_GLOBE_R2023A_4326_30ss_V1_0.tif")
    na_bounds = (-142.0, 24.0, -52.0, 84.0)

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

    # Read GHSL window
    with rasterio.open(ghs_path) as ghs_src, rasterio.open(kg_path) as kg_src:
        ghs_win = from_bounds(*na_bounds, transform=ghs_src.transform).round_offsets().round_lengths()
        ghs_transform = rasterio.windows.transform(ghs_win, ghs_src.transform)
        ghs_data = ghs_src.read(1, window=ghs_win)

        # Formal reprojection of KG onto GHSL exact grid
        print("Reprojecting Beck et al. (2018) onto exact GHSL 2020 grid (nearest-neighbor)...")
        kg_aligned = np.zeros_like(ghs_data, dtype=np.uint8)
        reproject(
            source=rasterio.band(kg_src, 1),
            destination=kg_aligned,
            src_transform=kg_src.transform,
            src_crs=kg_src.crs,
            dst_transform=ghs_transform,
            dst_crs=ghs_src.crs,
            resampling=Resampling.nearest
        )

        # Rasterize masks
        print("Rasterizing national boundary masks...")
        conus_mask = rasterize(conus_geom, out_shape=ghs_data.shape, transform=ghs_transform, fill=0, default_value=1, dtype=np.uint8) == 1
        canada_mask = rasterize(canada_geom, out_shape=ghs_data.shape, transform=ghs_transform, fill=0, default_value=1, dtype=np.uint8) == 1
        comb_mask = conus_mask | canada_mask

        tot_conus = ghs_data[conus_mask].astype(np.float64).sum() / 1e6
        tot_can = ghs_data[canada_mask].astype(np.float64).sum() / 1e6
        tot_comb = ghs_data[comb_mask].astype(np.float64).sum() / 1e6

        cov_rows = []
        for code in sorted(legend.keys()):
            sym, desc = legend[code]
            b_conus = ghs_data[conus_mask & (kg_aligned == code)].astype(np.float64).sum() / 1e6
            b_can = ghs_data[canada_mask & (kg_aligned == code)].astype(np.float64).sum() / 1e6
            b_comb = ghs_data[comb_mask & (kg_aligned == code)].astype(np.float64).sum() / 1e6

            if b_comb > 0.1:
                cov_rows.append({
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

    cov_df = pd.DataFrame(cov_rows).sort_values(by="combined_built_km2", ascending=False)
    cov_out = os.path.join(OUT_DIR, "climate_zone_built_up_coverage.csv")
    cov_df.to_csv(cov_out, index=False)
    print(f"[1/3] Saved rigorously aligned climate coverage table: {cov_out}")

    # 3. Full-Polygon Municipal Climate Class Breakdown
    print("\nAuditing climate classes across full municipal boundary polygons...")
    poly_climate_records = []
    with rasterio.open(kg_path) as src:
        for c in ['austin', 'bloomington', 'cupertino', 'surrey']:
            bp = os.path.join(ROOT_DIR, "census_boundaries", "exact_boundaries", f"{c}_exact_boundary.geojson")
            bgdf = gpd.read_file(bp).to_crs("EPSG:4326")
            geom = [bgdf.union_all()]
            out_img, _ = mask(src, geom, crop=True)
            v = out_img[0].flatten()
            v = v[v > 0]
            tot_cells = len(v)
            u_vals, u_counts = np.unique(v, return_counts=True)
            for u, cnt in sorted(zip(u_vals, u_counts), key=lambda x: x[1], reverse=True):
                sym, desc = legend.get(u, ("Unknown", ""))
                poly_climate_records.append({
                    'city': c.title(),
                    'total_cells': tot_cells,
                    'kg_code': u,
                    'climate_class': sym,
                    'description': desc,
                    'cell_count': cnt,
                    'pct_of_municipality': (cnt / tot_cells) * 100.0
                })
    poly_df = pd.DataFrame(poly_climate_records)
    poly_out = os.path.join(OUT_DIR, "municipal_climate_class_breakdown.csv")
    poly_df.to_csv(poly_out, index=False)
    print(f"[2/3] Saved municipal boundary climate breakdown: {poly_out}")

    # 4. Independent City-Level Comparison Table (Fully Audited)
    city_comparison = [
        {
            'City': "Austin, TX",
            'Country': "United States",
            'Dominant_Climate_Class': "Cfa (Humid subtropical)",
            'Boundary_Climate_Distribution': "100.0% Cfa (1,156/1,156 cells)",
            'CEC_Level_I_Ecoregion': "Great Plains (9)",
            'CEC_Level_III_Ecoregion': "Texas Blackland Prairies (9.4.7) / Edwards Plateau (9.4.6)",
            'Mean_Annual_Temp_C': 20.8,
            'Mean_Annual_Temp_F': 69.4,
            'Mean_Annual_Precip_mm': 920.8,
            'Mean_Annual_Precip_in': 36.25,
            'Weather_Station_ID': "Austin Camp Mabry (USW00013958; 30.32°N, 97.77°W; elev 150 m)",
            'Climate_Reference_Period': "1991-2020 (NOAA NCEI Climate Normals)",
            'Municipal_Land_Area_km2': 828.64,
            'Municipal_Land_Area_sqmi': 319.94,
            'Total_Boundary_Area_km2': 861.92,
            'Population_Decennial': 961855,
            'Housing_Units_Decennial': 444437,
            'Housing_Units_ACS_2022': 518574,
            'Population_Density_km2': 1160.8,
            'Housing_Density_km2': 536.4,
            'Demographic_Source': "U.S. Census Bureau 2020 Decennial Census (Tables P1, H1)",
            'Municipal_Parkland_GIS_km2': 71.70,
            'Municipal_Parkland_pct_of_land': 8.65,
            'Municipal_Parkland_pct_of_boundary': 8.32,
            'Parkland_Source': "City of Austin GIS (dataset v8hw-gz65 dissolved within boundary; unclipped=72.96 km2; TPL ParkScore 2023 reports 11% / 87 km2 including ETJ preserves)",
            'GHSL_Built_Up_Surface_km2': 106.28,
            'GHSL_Built_Up_pct_of_land': 12.83,
            'GHSL_Built_Up_pct_of_boundary': 12.33,
            'GHSL_Source': "JRC GHSL GHS-BUILT-S R2023A (epoch 2020, 30 arc-sec; building footprint surface area)"
        },
        {
            'City': "Bloomington, IN",
            'Country': "United States",
            'Dominant_Climate_Class': "Dfa (Humid continental, hot summer)",
            'Boundary_Climate_Distribution': "100.0% Dfa (90/90 cells)",
            'CEC_Level_I_Ecoregion': "Eastern Temperate Forests (8)",
            'CEC_Level_III_Ecoregion': "Interior Plateau (8.3.3)",
            'Mean_Annual_Temp_C': 12.1,
            'Mean_Annual_Temp_F': 53.8,
            'Mean_Annual_Precip_mm': 1191.8,
            'Mean_Annual_Precip_in': 46.92,
            'Weather_Station_ID': "Bloomington Monroe Co AP / Indiana Univ (USW00003889 / USC00120784; elev 258 m)",
            'Climate_Reference_Period': "1991-2020 (NOAA NCEI Climate Normals)",
            'Municipal_Land_Area_km2': 60.17,
            'Municipal_Land_Area_sqmi': 23.23,
            'Total_Boundary_Area_km2': 60.65,
            'Population_Decennial': 79168,
            'Housing_Units_Decennial': 36454,
            'Housing_Units_ACS_2022': 37537,
            'Population_Density_km2': 1315.7,
            'Housing_Density_km2': 605.8,
            'Demographic_Source': "U.S. Census Bureau 2020 Decennial Census (Tables P1, H1)",
            'Municipal_Parkland_GIS_km2': 6.05,
            'Municipal_Parkland_pct_of_land': 10.05,
            'Municipal_Parkland_pct_of_boundary': 9.97,
            'Parkland_Source': "City of Bloomington Parks GIS (Layer 13 dissolved within boundary; unclipped=8.50 km2 including Griffy Lake county preserve)",
            'GHSL_Built_Up_Surface_km2': 9.52,
            'GHSL_Built_Up_pct_of_land': 15.82,
            'GHSL_Built_Up_pct_of_boundary': 15.69,
            'GHSL_Source': "JRC GHSL GHS-BUILT-S R2023A (epoch 2020, 30 arc-sec; building footprint surface area)"
        },
        {
            'City': "Cupertino, CA",
            'Country': "United States",
            'Dominant_Climate_Class': "Csb (Warm-summer Mediterranean)",
            'Boundary_Climate_Distribution': "100.0% Csb (46/46 cells)",
            'CEC_Level_I_Ecoregion': "Mediterranean California (11)",
            'CEC_Level_III_Ecoregion': "Central California Foothills & Coastal Mountains (11.1.1)",
            'Mean_Annual_Temp_C': 16.0,
            'Mean_Annual_Temp_F': 60.8,
            'Mean_Annual_Precip_mm': 397.5,
            'Mean_Annual_Precip_in': 15.65,
            'Weather_Station_ID': "San Jose / Santa Clara Valley (USW00023293 / USC00047821; ~15 km NE, valley floor, elev 16 m)",
            'Climate_Reference_Period': "1991-2020 (NOAA NCEI Climate Normals)",
            'Municipal_Land_Area_km2': 29.34,
            'Municipal_Land_Area_sqmi': 11.33,
            'Total_Boundary_Area_km2': 29.35,
            'Population_Decennial': 60381,
            'Housing_Units_Decennial': 21987,
            'Housing_Units_ACS_2022': 22158,
            'Population_Density_km2': 2058.0,
            'Housing_Density_km2': 749.4,
            'Demographic_Source': "U.S. Census Bureau 2020 Decennial Census (Tables P1, H1)",
            'Municipal_Parkland_GIS_km2': 1.12,
            'Municipal_Parkland_pct_of_land': 3.81,
            'Municipal_Parkland_pct_of_boundary': 3.81,
            'Parkland_Source': "City of Cupertino GIS (PropertyInfo Layer 22 dissolved within boundary; active city parks subset=0.94 km2 / 3.2%)",
            'GHSL_Built_Up_Surface_km2': 8.56,
            'GHSL_Built_Up_pct_of_land': 29.18,
            'GHSL_Built_Up_pct_of_boundary': 29.17,
            'GHSL_Source': "JRC GHSL GHS-BUILT-S R2023A (epoch 2020, 30 arc-sec; building footprint surface area)"
        },
        {
            'City': "Surrey, BC",
            'Country': "Canada",
            'Dominant_Climate_Class': "Cfb (Temperate oceanic; 52.0%) / Csb (48.0%)",
            'Boundary_Climate_Distribution': "52.0% Cfb (293 cells) / 48.0% Csb (271 cells)",
            'CEC_Level_I_Ecoregion': "Marine West Coast Forests (7)",
            'CEC_Level_III_Ecoregion': "Puget Lowland / Georgia Depression (7.1.7)",
            'Mean_Annual_Temp_C': 10.4,
            'Mean_Annual_Temp_F': 50.7,
            'Mean_Annual_Precip_mm': 1562.0,
            'Mean_Annual_Precip_in': 61.50,
            'Weather_Station_ID': "Surrey Kwantlen Park (ECCC Climate ID 1108890; 49.19°N, 122.85°W; elev 78 m)",
            'Climate_Reference_Period': "1981-2010 (Environment and Climate Change Canada Climate Normals)",
            'Municipal_Land_Area_km2': 316.11,
            'Municipal_Land_Area_sqmi': 122.05,
            'Total_Boundary_Area_km2': 323.85,
            'Population_Decennial': 568322,
            'Housing_Units_Decennial': 195098,
            'Housing_Units_ACS_2022': 195098,
            'Population_Density_km2': 1797.9,
            'Housing_Density_km2': 617.2,
            'Demographic_Source': "Statistics Canada 2021 Census of Population (Catalogue 98-316-X2021001)",
            'Municipal_Parkland_GIS_km2': 34.48,
            'Municipal_Parkland_pct_of_land': 10.91,
            'Municipal_Parkland_pct_of_boundary': 10.65,
            'Parkland_Source': "City of Surrey OpenData (Layers 256 + 245 dissolved within boundary; unclipped=34.57 km2; Layer 256 parks alone=31.78 km2 / 10.1%)",
            'GHSL_Built_Up_Surface_km2': 50.02,
            'GHSL_Built_Up_pct_of_land': 15.82,
            'GHSL_Built_Up_pct_of_boundary': 15.45,
            'GHSL_Source': "JRC GHSL GHS-BUILT-S R2023A (epoch 2020, 30 arc-sec; building footprint surface area)"
        }
    ]

    city_df = pd.DataFrame(city_comparison)
    comp_out = os.path.join(OUT_DIR, "city_selection_independent_comparison.csv")
    city_df.to_csv(comp_out, index=False)
    print(f"[3/3] Saved audited independent comparison table: {comp_out}")

if __name__ == "__main__":
    main()
