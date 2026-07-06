import pandas as pd
import geopandas as gpd
import os

TREES_DIR = os.environ.get("TREE_MAPPING_TREES_DIR", os.path.join(os.environ.get("TREE_MAPPING_BASE_DIR", os.getcwd()), "all_trees"))
CITIES = ['austin', 'bloomington', 'cupertino', 'surrey']

def main():
    for city in CITIES:
        csv_path = os.path.join(TREES_DIR, f"{city}_ripley_points_with_income.csv")
        output_geojson = os.path.join(TREES_DIR, f"{city}_ripley_points_with_income.geojson")
        
        if not os.path.exists(csv_path):
            print(f"Skipping {city}: CSV not found.")
            continue
            
        print(f"Converting {city} CSV to GeoJSON...")
        df = pd.read_csv(csv_path)
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.longitude, df.latitude),
            crs="EPSG:4326"
        )
        
        # Drop longitude/latitude columns if desired (optional, as they are now in geometry)
        # However, keeping them matches the "match the columns" intent if they were properties.
        # But in GeoJSON, geometry is separate. I'll keep them as properties for redundancy.
        
        # Save to GeoJSON
        print(f"  Saving {len(gdf):,} features to {output_geojson}...")
        gdf.to_file(output_geojson, driver='GeoJSON')
        print(f"  Done.")

if __name__ == "__main__":
    main()
