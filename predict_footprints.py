import os
import geopandas as gpd
import pandas as pd
import joblib

def predict_footprints():
    data_dir = r"C:\Users\Angelica\OneDrive\Apartment Analysis\data"
    input_geojson = os.path.join(data_dir, "multifamily_master.geojson") 
    output_geojson = os.path.join(data_dir, "multifamily_predictions.geojson")
    models_dir = os.path.join(data_dir, "models")
    
    if not os.path.exists(input_geojson):
        print(f"ERROR: Could not find input file: {input_geojson}")
        return

    print(f"Loading spatial data from: {input_geojson}...")
    gdf = gpd.read_file(input_geojson)
    gdf.columns = [c.replace('properties/', '') for c in gdf.columns]

    print("Preparing features for prediction...")
    prep_df = pd.DataFrame()
    prep_df['parcel_width'] = pd.to_numeric(gdf['estimated_lot_width'], errors='coerce').fillna(50.0)
    prep_df['parcel_area'] = pd.to_numeric(gdf['ll_gissqft'], errors='coerce').fillna(6000.0)
    prep_df['parcel_depth'] = prep_df['parcel_area'] / prep_df['parcel_width']
    prep_df['year_built'] = pd.to_numeric(gdf['YR_BLT'], errors='coerce').fillna(1980)
    prep_df['zoning_district'] = gdf['zoning'].astype(str).fillna('Unknown')
    prep_df['context'] = gdf['Context'].astype(str).fillna('Unknown')
    
    features = ['parcel_width', 'parcel_depth', 'year_built', 'zoning_district', 'context']
    X = prep_df[features]

    # Every single targeted metric we want the script to evaluate
    all_models = ['height', 'coverage', 'footprint', 'width', 'depth', 'shape']
    
    print("\n--- Running AI Parameter & Shape Predictions ---")
    for model_name in all_models:
        model_file = os.path.join(models_dir, f'footprint_{model_name}_model.joblib')
        
        if not os.path.exists(model_file):
            print(f"Warning: Model file not found, skipping: {model_file}")
            continue
            
        print(f"Loading {model_name} model...")
        model = joblib.load(model_file)
        
        # Attach prediction vectors straight to the geographical dataframe
        gdf[f'pred_{model_name}'] = model.predict(X)
        print(f"-> Successfully predicted: pred_{model_name}")

    print(f"\nSaving detailed prediction layers to: {output_geojson}...")
    gdf.to_file(output_geojson, driver='GeoJSON')
    print("Process complete! All structural metrics have been populated.")

if __name__ == "__main__":
    predict_footprints()