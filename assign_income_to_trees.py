import os
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = os.environ.get("TREE_MAPPING_BASE_DIR", os.getcwd())
TREES_DIR = os.environ.get("TREE_MAPPING_TREES_DIR", os.path.join(os.environ.get("TREE_MAPPING_BASE_DIR", os.getcwd()), "all_trees"))
CENSUS_DIR = os.path.join(BASE_DIR, "notebooks/census_boundaries")
INCOME_DIR = os.path.join(BASE_DIR, "notebooks/income_data")

CITIES = ['austin', 'bloomington', 'cupertino', 'surrey']

CITY_CONFIG = {
    'austin': {
        'utm_epsg': 32614,
        'census_path': os.path.join(CENSUS_DIR, 'tl_2023_48_bg/tl_2023_48_bg.shp'),
        'filter_col': 'COUNTYFP',
        'filter_val': '453',
        'geoid_col': 'GEOID',
        'country': 'US'
    },
    'bloomington': {
        'utm_epsg': 32616,
        'census_path': os.path.join(CENSUS_DIR, 'tl_2023_18_bg/tl_2023_18_bg.shp'),
        'filter_col': 'COUNTYFP',
        'filter_val': '105',
        'geoid_col': 'GEOID',
        'country': 'US'
    },
    'cupertino': {
        'utm_epsg': 32610,
        'census_path': os.path.join(CENSUS_DIR, 'tl_2023_06_bg/tl_2023_06_bg.shp'),
        'filter_col': 'COUNTYFP',
        'filter_val': '085',
        'geoid_col': 'GEOID',
        'country': 'US'
    },
    'surrey': {
        'utm_epsg': 32610,
        'census_path': os.path.join(CENSUS_DIR, 'lda_000b21a_e/lda_000b21a_e.shp'),
        'filter_col': 'CSDUID',
        'filter_val': '5915004',
        'geoid_col': 'DAUID',
        'country': 'CA'
    }
}

US_INCOME_PATH = os.path.join(INCOME_DIR, 'ACSDT5Y2023/ACSDT5Y2023.B19013-Data.csv')
CA_INCOME_PATH = os.path.join(INCOME_DIR, '98-401-X2021006_BC_CB_eng_CSV/98-401-X2021006_English_CSV_data_BritishColumbia.csv')

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_us_income_data(filepath):
    """Load and clean US ACS median household income data."""
    print(f"Loading US income data from {filepath}...")
    df = pd.read_csv(filepath, skiprows=1, dtype={'GEO_ID': str})
    
    # Clean column names
    if len(df.columns) == 5:
        df.columns = ['GEOID_full', 'NAME', 'median_income', 'income_moe', '_empty']
        df = df.drop(columns=['_empty'])
    else:
        df.columns = ['GEOID_full', 'NAME', 'median_income', 'income_moe']
    
    # Extract 12-digit GEOID
    df['GEOID'] = df['GEOID_full'].str.replace('1500000US', '', regex=False)
    df['median_income'] = pd.to_numeric(df['median_income'], errors='coerce')
    
    return df[['GEOID', 'median_income']].dropna(subset=['median_income'])

def load_canadian_income_data(filepath, region_prefix='59'):
    """Load Canadian Census 2021 median household income (Characteristic ID 243)."""
    print(f"Loading Canadian income data from {filepath}...")
    cols_to_use = ['ALT_GEO_CODE', 'GEO_LEVEL', 'GEO_NAME', 'CHARACTERISTIC_ID', 'C1_COUNT_TOTAL']
    
    chunks = []
    # Statistics Canada files often use latin-1
    for chunk in pd.read_csv(filepath, usecols=cols_to_use, chunksize=500000, 
                             dtype={'ALT_GEO_CODE': str, 'CHARACTERISTIC_ID': int},
                             encoding='latin-1'):
        mask = (chunk['GEO_LEVEL'] == 'Dissemination area') & (chunk['CHARACTERISTIC_ID'] == 243)
        if region_prefix:
            mask = mask & chunk['ALT_GEO_CODE'].str.startswith(region_prefix)
        
        filtered = chunk[mask].copy()
        if not filtered.empty:
            filtered['C1_COUNT_TOTAL'] = pd.to_numeric(filtered['C1_COUNT_TOTAL'], errors='coerce')
            chunks.append(filtered)
            
    if not chunks: return pd.DataFrame()
    
    df = pd.concat(chunks)
    return df.rename(columns={'ALT_GEO_CODE': 'DAUID', 'C1_COUNT_TOTAL': 'median_income'})[['DAUID', 'median_income']].dropna()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    # Pre-load income data
    us_income = load_us_income_data(US_INCOME_PATH)
    ca_income = load_canadian_income_data(CA_INCOME_PATH)
    
    for city in CITIES:
        print(f"\nProcessing {city.upper()}...")
        cfg = CITY_CONFIG[city]
        
        # 1. Load and Filter Census Boundaries (Keeping all to ensure full coverage)
        print(f"  Loading census boundaries...")
        gdf_census = gpd.read_file(cfg['census_path'])
        # (Filtering removed to ensure trees on city/county edges are still assigned a block group)
        
        if gdf_census.crs != 'EPSG:4326':
            gdf_census = gdf_census.to_crs('EPSG:4326')
            
        # 2. Join Income Data to Census (Left join to keep all boundaries)
        print(f"  Joining income data...")
        income_df = us_income if cfg['country'] == 'US' else ca_income
        gdf_census = gdf_census.merge(income_df, left_on=cfg['geoid_col'], right_on=cfg['geoid_col'], how='left')
        
        if gdf_census.empty:
            print(f"  Warning: No census units with income found for {city}. Check GEOID matching.")
            continue
            
        # 3. Load Tree Points
        csv_path = os.path.join(TREES_DIR, f"{city}_ripley_points_final.csv")
        if not os.path.exists(csv_path):
            print(f"  Warning: Tree CSV not found: {csv_path}")
            continue
            
        print(f"  Loading {len(gdf_census)} census units and tree points...")
        df_trees = pd.read_csv(csv_path)
        gdf_trees = gpd.GeoDataFrame(
            df_trees,
            geometry=gpd.points_from_xy(df_trees.x_meters, df_trees.y_meters),
            crs=f"EPSG:{cfg['utm_epsg']}"
        ).to_crs("EPSG:4326")
        
        # 4. Spatial Join (Trees -> Census)
        print(f"  Performing spatial join ({len(gdf_trees):,} trees)...")
        # Left join to keep all trees, even those without income or outside boundaries
        joined = gpd.sjoin(gdf_trees, gdf_census[[cfg['geoid_col'], 'median_income', 'geometry']], 
                           how='left', predicate='intersects')
        
        # 1:1 assignment for trees on boundaries
        joined = joined[~joined.index.duplicated(keep='first')]
        
        # 5. Format to match GeoJSON schema
        joined['city'] = city
        joined['longitude'] = joined.geometry.x
        joined['latitude'] = joined.geometry.y
        
        # Add has_income column
        joined['has_income'] = np.where(joined['median_income'].isna(), 'no', 'yes')
        
        # Rename columns to match GeoJSON properties
        joined = joined.rename(columns={
            cfg['geoid_col']: 'block_group_id',
            'median_income': 'median_household_income'
        })
        
        # Select and order columns
        final_cols = ['city', 'block_group_id', 'median_household_income', 'has_income', 
                      'longitude', 'latitude', 'x_meters', 'y_meters', 'crown_area_px']
        df_output = joined[final_cols]
        
        # 6. Save Results
        output_csv = os.path.join(TREES_DIR, f"{city}_ripley_points_with_income.csv")
        df_output.to_csv(output_csv, index=False)
        print(f"  Successfully saved {len(df_output):,} trees with income to {output_csv}")

if __name__ == "__main__":
    main()
