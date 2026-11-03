import os
import json
import pandas as pd
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================
ROOT = os.environ.get("URBAN_DATA", "/path/to/Urban_data")
BASE_DIR = os.path.join(ROOT, "sherlock_venv")
INCOME_DIR = os.path.join(BASE_DIR, "notebooks/income_data")
TREES_DIR = os.path.join(ROOT, "ripley_data_0429206/all_trees")

US_B03002_PATH = os.path.join(INCOME_DIR, "ACSDT5Y2023.B03002_2026-06-02T143300/ACSDT5Y2023.B03002-Data.csv")
CA_CENSUS_PATH = os.path.join(INCOME_DIR, "98-401-X2021006_BC_CB_eng_CSV/98-401-X2021006_English_CSV_data_BritishColumbia.csv")

CITIES = {
    'cupertino': {'country': 'US', 'geojson': 'cupertino_ripley_points_with_income.geojson'},
    'bloomington': {'country': 'US', 'geojson': 'bloomington_ripley_points_with_income.geojson'},
    'austin': {'country': 'US', 'geojson': 'austin_ripley_points_with_income.geojson'},
    'surrey': {'country': 'CA', 'geojson': 'surrey_ripley_points_with_income.geojson'}
}

# ============================================================================
# STEP 1: LOAD AND PROCESS US ACS DEMOGRAPHICS (B03002)
# ============================================================================
def load_us_demographics():
    print(f"Loading US demographics from {US_B03002_PATH}...")
    # Skip the second descriptive row (index 1)
    df = pd.read_csv(US_B03002_PATH, skiprows=[1])
    
    # Extract 12-digit GEOID (stripping the Geography identifier prefix)
    df['GEOID'] = df['GEO_ID'].str.replace('1500000US', '', regex=False)
    
    # Select demographic columns and convert to numeric
    cols = {
        'total_pop': 'B03002_001E',
        'white_alone': 'B03002_003E',
        'black_alone': 'B03002_004E',
        'native_amer': 'B03002_005E',
        'asian_alone': 'B03002_006E',
        'hawaiian_pi': 'B03002_007E',
        'other_alone': 'B03002_008E',
        'two_more': 'B03002_009E',
        'hispanic': 'B03002_012E'
    }
    
    for key, col in cols.items():
        df[key] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Calculate standardized percentages
    # Handle zero division safely by leaving values as NaN (which will map to None in geojson)
    total = df['total_pop']
    valid_mask = total > 0
    
    df['pct_white'] = np.where(valid_mask, df['white_alone'] / total, np.nan)
    df['pct_black'] = np.where(valid_mask, df['black_alone'] / total, np.nan)
    df['pct_asian'] = np.where(valid_mask, df['asian_alone'] / total, np.nan)
    df['pct_hispanic_latino'] = np.where(valid_mask, df['hispanic'] / total, np.nan)
    df['pct_indigenous'] = np.where(valid_mask, (df['native_amer'] + df['hawaiian_pi']) / total, np.nan)
    df['pct_other_multiracial'] = np.where(valid_mask, (df['other_alone'] + df['two_more']) / total, np.nan)
    
    # Normalize to sum to exactly 1.0 (corrects for float precision/rounding)
    demo_sum = (df['pct_white'].fillna(0) + df['pct_black'].fillna(0) + df['pct_asian'].fillna(0) + 
                df['pct_hispanic_latino'].fillna(0) + df['pct_indigenous'].fillna(0) + df['pct_other_multiracial'].fillna(0))
    sum_mask = demo_sum > 0
    df['pct_white'] = np.where(sum_mask, df['pct_white'] / demo_sum, df['pct_white'])
    df['pct_black'] = np.where(sum_mask, df['pct_black'] / demo_sum, df['pct_black'])
    df['pct_asian'] = np.where(sum_mask, df['pct_asian'] / demo_sum, df['pct_asian'])
    df['pct_hispanic_latino'] = np.where(sum_mask, df['pct_hispanic_latino'] / demo_sum, df['pct_hispanic_latino'])
    df['pct_indigenous'] = np.where(sum_mask, df['pct_indigenous'] / demo_sum, df['pct_indigenous'])
    df['pct_other_multiracial'] = np.where(sum_mask, df['pct_other_multiracial'] / demo_sum, df['pct_other_multiracial'])

    # Create lookup dictionary (key converted to str(int(geoid)) to handle potential leading zero mismatch)
    lookup = {}
    for _, row in df.iterrows():
        try:
            geoid_key = str(int(row['GEOID']))
            lookup[geoid_key] = {
                'total_pop': int(row['total_pop']) if not pd.isna(row['total_pop']) else None,
                'pct_white': float(row['pct_white']) if not pd.isna(row['pct_white']) else None,
                'pct_black': float(row['pct_black']) if not pd.isna(row['pct_black']) else None,
                'pct_asian': float(row['pct_asian']) if not pd.isna(row['pct_asian']) else None,
                'pct_hispanic_latino': float(row['pct_hispanic_latino']) if not pd.isna(row['pct_hispanic_latino']) else None,
                'pct_indigenous': float(row['pct_indigenous']) if not pd.isna(row['pct_indigenous']) else None,
                'pct_other_multiracial': float(row['pct_other_multiracial']) if not pd.isna(row['pct_other_multiracial']) else None
            }
        except Exception as e:
            continue
            
    print(f"Loaded {len(lookup)} US block group demographic profiles.")
    return lookup

# ============================================================================
# STEP 2: LOAD AND PROCESS CANADIAN DEMOGRAPHICS
# ============================================================================
def load_ca_demographics():
    print(f"Loading Canadian demographics from {CA_CENSUS_PATH}...")
    char_ids = [1403, 1683, 1685, 1686, 1687, 1688, 1689, 1690, 1691, 1692, 1693, 1694, 1695, 1696, 1697]
    
    chunks = []
    # Read in chunks due to size (3.6 GB)
    for chunk in pd.read_csv(CA_CENSUS_PATH, usecols=['ALT_GEO_CODE', 'GEO_LEVEL', 'CHARACTERISTIC_ID', 'C1_COUNT_TOTAL'],
                             chunksize=1000000, dtype={'ALT_GEO_CODE': str, 'CHARACTERISTIC_ID': int}, encoding='latin-1'):
        # Filter for Surrey dissemination areas (starts with 5915)
        mask = (chunk['GEO_LEVEL'] == 'Dissemination area') & \
               (chunk['ALT_GEO_CODE'].str.startswith('5915')) & \
               (chunk['CHARACTERISTIC_ID'].isin(char_ids))
        
        filtered = chunk[mask].copy()
        if not filtered.empty:
            chunks.append(filtered)
            
    if not chunks:
        print("No Canadian demographic data found.")
        return {}
        
    df_all = pd.concat(chunks)
    df_all['C1_COUNT_TOTAL'] = pd.to_numeric(df_all['C1_COUNT_TOTAL'], errors='coerce').fillna(0)
    
    # Pivot long format to wide
    df_wide = df_all.pivot(index='ALT_GEO_CODE', columns='CHARACTERISTIC_ID', values='C1_COUNT_TOTAL').reset_index()
    
    # Rename columns to their actual meaning for readability in calculations
    cols = {
        'dauid': 'ALT_GEO_CODE',
        'indigenous': 1403,
        'total_pop': 1683,
        'south_asian': 1685,
        'chinese': 1686,
        'black': 1687,
        'filipino': 1688,
        'arab': 1689,
        'latin_amer': 1690,
        'se_asian': 1691,
        'west_asian': 1692,
        'korean': 1693,
        'japanese': 1694,
        'visible_nie': 1695,
        'visible_multiple': 1696,
        'not_visible': 1697
    }
    
    for name, cid in cols.items():
        if cid in df_wide.columns:
            df_wide[name] = df_wide[cid]
        elif name != 'dauid':
            df_wide[name] = 0.0 # fallback if missing
            
    total = df_wide['total_pop']
    valid_mask = total > 0
    
    # Calculate standardized percentages
    # Move Arab and West Asian to pct_white to align with U.S. ACS standards (White alone)
    white_sum_ca = df_wide['not_visible'] - df_wide['indigenous'] + df_wide['arab'] + df_wide['west_asian']
    df_wide['pct_white'] = np.where(valid_mask, white_sum_ca / total, np.nan)
    df_wide['pct_black'] = np.where(valid_mask, df_wide['black'] / total, np.nan)
    
    asian_sum = (df_wide['south_asian'] + df_wide['chinese'] + df_wide['filipino'] + 
                 df_wide['se_asian'] + df_wide['korean'] + df_wide['japanese'])
    df_wide['pct_asian'] = np.where(valid_mask, asian_sum / total, np.nan)
    df_wide['pct_hispanic_latino'] = np.where(valid_mask, df_wide['latin_amer'] / total, np.nan)
    df_wide['pct_indigenous'] = np.where(valid_mask, df_wide['indigenous'] / total, np.nan)
    
    # Exclude arab and west_asian from other/multiracial
    other_sum = (df_wide['visible_nie'] + df_wide['visible_multiple'])
    df_wide['pct_other_multiracial'] = np.where(valid_mask, other_sum / total, np.nan)
    
    # Normalize to sum to exactly 1.0 (corrects for Statistics Canada's independent random rounding at the DA level)
    demo_sum = (df_wide['pct_white'].fillna(0) + df_wide['pct_black'].fillna(0) + df_wide['pct_asian'].fillna(0) + 
                df_wide['pct_hispanic_latino'].fillna(0) + df_wide['pct_indigenous'].fillna(0) + df_wide['pct_other_multiracial'].fillna(0))
    sum_mask = demo_sum > 0
    df_wide['pct_white'] = np.where(sum_mask, df_wide['pct_white'] / demo_sum, df_wide['pct_white'])
    df_wide['pct_black'] = np.where(sum_mask, df_wide['pct_black'] / demo_sum, df_wide['pct_black'])
    df_wide['pct_asian'] = np.where(sum_mask, df_wide['pct_asian'] / demo_sum, df_wide['pct_asian'])
    df_wide['pct_hispanic_latino'] = np.where(sum_mask, df_wide['pct_hispanic_latino'] / demo_sum, df_wide['pct_hispanic_latino'])
    df_wide['pct_indigenous'] = np.where(sum_mask, df_wide['pct_indigenous'] / demo_sum, df_wide['pct_indigenous'])
    df_wide['pct_other_multiracial'] = np.where(sum_mask, df_wide['pct_other_multiracial'] / demo_sum, df_wide['pct_other_multiracial'])

    # Create lookup dictionary (key converted to str(int(dauid)) to handle leading zero mismatch/types)
    lookup = {}
    for _, row in df_wide.iterrows():
        try:
            dauid_key = str(int(row['dauid']))
            lookup[dauid_key] = {
                'total_pop': int(row['total_pop']) if not pd.isna(row['total_pop']) else None,
                'pct_white': float(row['pct_white']) if not pd.isna(row['pct_white']) else None,
                'pct_black': float(row['pct_black']) if not pd.isna(row['pct_black']) else None,
                'pct_asian': float(row['pct_asian']) if not pd.isna(row['pct_asian']) else None,
                'pct_hispanic_latino': float(row['pct_hispanic_latino']) if not pd.isna(row['pct_hispanic_latino']) else None,
                'pct_indigenous': float(row['pct_indigenous']) if not pd.isna(row['pct_indigenous']) else None,
                'pct_other_multiracial': float(row['pct_other_multiracial']) if not pd.isna(row['pct_other_multiracial']) else None
            }
        except Exception as e:
            continue
            
    print(f"Loaded {len(lookup)} Canadian DA demographic profiles.")
    return lookup

# ============================================================================
# STEP 3: GEOJSON INTEGRATION
# ============================================================================
def update_geojson(city, config, demographics_lookup):
    geojson_filename = config['geojson']
    in_path = os.path.join(TREES_DIR, geojson_filename)
    out_path = os.path.join(TREES_DIR, geojson_filename.replace('.geojson', '_temp.geojson'))
    
    if not os.path.exists(in_path):
        print(f"Skipping {city}: file not found at {in_path}")
        return
        
    print(f"Processing {city.upper()} ({geojson_filename})...")
    
    matched_count = 0
    missing_count = 0
    total_features = 0
    
    # Open input and output files for line-by-line processing
    with open(in_path, 'r') as fin, open(out_path, 'w') as fout:
        for line in fin:
            stripped = line.strip()
            # If the line represents a feature, process it
            if stripped.startswith('{"type": "Feature"') or stripped.startswith('{ "type": "Feature"'):
                total_features += 1
                if total_features % 100000 == 0:
                    print(f"  Processed {total_features:,} features...")
                
                # Check for trailing comma and strip it to parse JSON
                has_comma = stripped.endswith(',')
                json_str = stripped[:-1] if has_comma else stripped
                
                try:
                    feature = json.loads(json_str)
                    props = feature.get('properties', {})
                    block_group_id = props.get('block_group_id')
                    
                    demo_profile = None
                    if block_group_id is not None:
                        # Convert to normalized string representation of integer for matching
                        bg_key = str(int(float(block_group_id)))
                        demo_profile = demographics_lookup.get(bg_key)
                    
                    # Update properties with demographic data
                    if demo_profile:
                        props.update(demo_profile)
                        matched_count += 1
                    else:
                        props.update({
                            'total_pop': None,
                            'pct_white': None,
                            'pct_black': None,
                            'pct_asian': None,
                            'pct_hispanic_latino': None,
                            'pct_indigenous': None,
                            'pct_other_multiracial': None
                        })
                        missing_count += 1
                        
                    feature['properties'] = props
                    
                    # Write updated feature back
                    new_line = json.dumps(feature)
                    if has_comma:
                        new_line += ',\n'
                    else:
                        new_line += '\n'
                    fout.write(new_line)
                    
                except Exception as e:
                    # If parsing fails, write the line back unmodified
                    fout.write(line)
            else:
                # Header/footer metadata lines, write unmodified
                fout.write(line)
                
    # Replace original file with the updated temp file
    os.replace(out_path, in_path)
    print(f"Finished {city.upper()}. Matched: {matched_count:,}, Missing/No Census: {missing_count:,}, Total: {total_features:,}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    us_demo = load_us_demographics()
    ca_demo = load_ca_demographics()
    
    for city, config in CITIES.items():
        lookup = us_demo if config['country'] == 'US' else ca_demo
        update_geojson(city, config, lookup)
        
    print("\nAll GeoJSON files successfully updated with standardized racial demographics!")

if __name__ == "__main__":
    main()
