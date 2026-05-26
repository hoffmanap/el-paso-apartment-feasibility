import os
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
import joblib

def extract_mapped_features(geojson_path):
    """
    Parses the GeoJSON using the exact schema from your dataset.
    Dynamically calculates missing geometry fields (Width, Depth, Shape, FAR).
    """
    print(f"Loading spatial data from: {geojson_path}...")
    
    if not os.path.exists(geojson_path):
        print(f"ERROR: Could not find {geojson_path}.")
        return pd.DataFrame()
        
    gdf = gpd.read_file(geojson_path)
    
    # Normalize column names by removing any nested 'properties/' prefix strings
    gdf.columns = [c.replace('properties/', '') for c in gdf.columns]
    
    # Project to feet BEFORE doing any structural loops or dimension filtering
    if gdf.crs is None:
        print("No CRS detected. Explicitly assigning WGS84 (Lat/Lon)...")
        gdf.set_crs(epsg=4326, inplace=True)
        
    print("Reprojecting coordinates to Texas Feet-based Coordinate System (EPSG:2277)...")
    gdf = gdf.to_crs(epsg=2277) 
    
    records = []
    
    print("Calculating building physical parameters and configuration classifications...")
    for idx, row in gdf.iterrows():
        # --- INPUT FEATURES (X) ---
        parcel_w = float(pd.to_numeric(row.get('estimated_lot_width'), errors='coerce') if pd.notnull(row.get('estimated_lot_width')) else 50.0)
        parcel_area = float(pd.to_numeric(row.get('ll_gissqft'), errors='coerce') if pd.notnull(row.get('ll_gissqft')) else 6000.0)
        
        # Calculate parcel depth (Area / Width)
        parcel_d = parcel_area / parcel_w if parcel_w > 0 else 120.0
        
        zoning = str(row.get('zoning')).strip() if pd.notnull(row.get('zoning')) else 'Unknown'
        context = str(row.get('Context')).strip() if pd.notnull(row.get('Context')) else 'Unknown'
        
        # Safely convert YR_BLT to numeric first, handle NaNs, then cast to integer safely
        raw_year = pd.to_numeric(row.get('YR_BLT'), errors='coerce')
        year = int(raw_year) if pd.notnull(raw_year) else 1980
        
        # --- TARGET PREDICTIONS (y) ---
        height = float(pd.to_numeric(row.get('HEIGHT_FT'), errors='coerce') if pd.notnull(row.get('HEIGHT_FT')) else 0.0)
        coverage = float(pd.to_numeric(row.get('lot_coverage'), errors='coerce') if pd.notnull(row.get('lot_coverage')) else 0.0)
        
        # Find building footprint area safely using standard dataset properties
        if 'SQFEET' in row and pd.notnull(row.get('SQFEET')):
            bldg_sqft = float(row.get('SQFEET'))
        elif 'Shape_Area' in row and pd.notnull(row.get('Shape_Area')):
            bldg_sqft = float(row.get('Shape_Area'))
        else:
            bldg_sqft = float(row.get('geometry').area) if pd.notnull(row.get('geometry')) else 0.0
            
        # Calculate Floor Area Ratio (FAR) = Gross SqFt / Lot SqFt
        far = (bldg_sqft / parcel_area) if parcel_area > 0 else 0.0
        
        # --- GEOMETRIC EXTRACTION ---
        building_poly = row.get('geometry', None)
        bldg_w, bldg_d = 0.0, 0.0
        shape_type = 'Rectangle / Box'
        
        # Ensure the polygon exists and isn't empty before performing geometric math
        if pd.notnull(building_poly) and hasattr(building_poly, 'is_empty') and not building_poly.is_empty:
            
            # Unpack MultiPolygons safely vs standard Polygons using explicit string types
            polygons = list(building_poly.geoms) if building_poly.geom_type == 'MultiPolygon' else [building_poly]
            
            # Use len(p.interiors) instead of num_interior_rings to find courtyards safely
            has_courtyard = any(len(p.interiors) > 0 for p in polygons if hasattr(p, 'interiors'))
            
            # Initialize bounding box attributes in true projected FEET
            rot_rect = building_poly.minimum_rotated_rectangle
            
            if rot_rect.geom_type == 'Polygon' and hasattr(rot_rect, 'exterior') and rot_rect.exterior is not None:
                try:
                    x, y = rot_rect.exterior.coords.xy
                    
                    # Native ZIP mapping to determine edge distances without array dimension failures
                    edge_lengths = [
                        ((x1 - x2)**2 + (y1 - y2)**2)**0.5
                        for (x1, y1), (x2, y2) in zip(zip(x, y), zip(x[1:], y[1:]))
                    ]
                    
                    if edge_lengths:
                        bldg_w = max(edge_lengths)
                        bldg_d = min(edge_lengths)
                    else:
                        raise ValueError("No edges calculated")
                except:
                    bounds = building_poly.bounds  
                    bldg_w = bounds[2] - bounds[0]
                    bldg_d = bounds[3] - bounds[1]
            else:
                bounds = building_poly.bounds
                bldg_w = bounds[2] - bounds[0]
                bldg_d = bounds[3] - bounds[1]
            
            # Normalize fallback default dimensional limits (now accurately working in feet!)
            bldg_w = bldg_w if bldg_w > 5.0 else 30.0
            bldg_d = bldg_d if bldg_d > 5.0 else 40.0
            
            # Determine Shape Typology based on how "solid" the footprint is
            try:
                convex_area = building_poly.convex_hull.area
                if convex_area > 0:
                    solidity = building_poly.area / convex_area
                    if has_courtyard or solidity < 0.60:
                        shape_type = 'Complex / Courtyard'
                    elif solidity > 0.90:
                        shape_type = 'Rectangle / Box'
                    else:
                        shape_type = 'L-Shape / T-Shape'
            except:
                pass

        # Skip rows that lack realistic building geometry sizes
        if bldg_w <= 5.0 or bldg_d <= 5.0:
            continue

        records.append({
            'parcel_width': parcel_w,
            'parcel_depth': parcel_d,
            'zoning_district': zoning,
            'year_built': year,
            'context': context,
            
            'target_width': bldg_w,
            'target_depth': bldg_d,
            'target_shape': shape_type,
            'target_coverage': coverage,
            'target_far': far,
            'target_height': height,
            'target_footprint': bldg_sqft
        })

    df = pd.DataFrame(records)
    print(f"Successfully processed {len(df)} compliant records from your data!")
    return df

def execute_model_training():
    data_dir = r"C:\Users\Angelica\OneDrive\Apartment Analysis\data"
    geojson_path = os.path.join(data_dir, "multifamily_master.geojson") 
    models_dir = os.path.join(data_dir, "models")
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    df = extract_mapped_features(geojson_path)
    if df.empty:
        print("Training aborted: No valid data available.")
        return

    # Explicit layout settings
    features = ['parcel_width', 'parcel_depth', 'year_built', 'zoning_district', 'context']
    numeric_features = ['parcel_width', 'parcel_depth', 'year_built']
    categorical_features = ['zoning_district', 'context']

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    print("\n--- Training & Evaluating AI Models ---")
    regression_targets = {
        'width': 'target_width',
        'depth': 'target_depth',
        'coverage': 'target_coverage',
        'far': 'target_far',
        'height': 'target_height',
        'footprint': 'target_footprint'
    }

    # 1. Train continuous regression models
    for model_name, target_col in regression_targets.items():
        clean_df = df.dropna(subset=[target_col]).copy()
        
        # Ensure categorical inputs are string formatted so OneHotEncoder never stumbles on empty/numeric values
        for cat in categorical_features:
            clean_df[cat] = clean_df[cat].astype(str).fillna('Unknown')
            
        if clean_df.empty:
            print(f"Skipping model {model_name.upper()}: No valid samples remaining.")
            continue
            
        X_clean = clean_df[features]
        y = clean_df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X_clean, y, test_size=0.2, random_state=42)
        
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('algo', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        
        pipeline.fit(X_train, y_train)
        
        preds = pipeline.predict(X_test)
        print(f"Model [ {model_name.upper()} ] -> MAE: {mean_absolute_error(y_test, preds):.2f}, R²: {r2_score(y_test, preds):.2f}")
        
        joblib.dump(pipeline, os.path.join(models_dir, f'footprint_{model_name}_model.joblib'))

    # 2. Train categorical shape classification model
    print("\nTraining shape classification model...")
    clean_df = df.dropna(subset=['target_shape']).copy()
    
    for cat in categorical_features:
        clean_df[cat] = clean_df[cat].astype(str).fillna('Unknown')
        
    if not clean_df.empty:
        X_clean = clean_df[features]
        y_shape = clean_df['target_shape']
        
        X_train, X_test, y_train, y_test = train_test_split(X_clean, y_shape, test_size=0.2, random_state=42)
        
        shape_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('algo', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        shape_pipeline.fit(X_train, y_train)
        
        shape_preds = shape_pipeline.predict(X_test)
        print(f"Model [ SHAPE CLASSIFIER ] -> Accuracy: {accuracy_score(y_test, shape_preds):.2%}")
        
        joblib.dump(shape_pipeline, os.path.join(models_dir, 'footprint_shape_model.joblib'))
    else:
        print("Skipping SHAPE CLASSIFIER: No valid structural classification samples remaining.")
    
    print(f"\nSUCCESS! All models successfully built and evaluated!")

# --- EXECUTION HOOK (Ensures the script executes upon call) ---
if __name__ == "__main__":
    execute_model_training()