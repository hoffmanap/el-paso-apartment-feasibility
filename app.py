import os
import streamlit as st
import geopandas as gpd
import folium
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from streamlit_folium import st_folium

# 1. Set up the web page configurations
st.set_page_config(page_title="El Paso Apartment Feasibility AI", layout="wide")

st.title("🏢 El Paso Apartment Design & Feasibility AI")
st.markdown("Explore existing multifamily parcels and look at predicted AI massing variables based on zoning, context, and lot features.")

# 2. Path to your prediction outputs (RELATIVE FOR CLOUD HOSTING)
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
prediction_file = os.path.join(current_dir, "data", "multifamily_predictions.geojson")

@st.cache_data
def load_web_data(path):
    if not os.path.exists(path):
        return None
    gdf = gpd.read_file(path)
    
    # Standardize ALL columns to lowercase right out of the gate
    gdf.columns = [c.replace('properties/', '').lower().strip() for c in gdf.columns]
    return gdf.to_crs(epsg=4326)

gdf = load_web_data(prediction_file)

if gdf is None:
    st.error(f"Could not find the predictions file at {prediction_file}. Please make sure you've successfully run predict_footprints.py first and your 'data' folder is pushed to GitHub!")
else:
    # --- HARDCODED STABLE COLUMN CONFIGURATIONS ---
    zoning_col = 'zoning' if 'zoning' in gdf.columns else gdf.columns[0]
    address_col = 'address' if 'address' in gdf.columns else gdf.columns[1]
    year_col = 'yr_blt' if 'yr_blt' in gdf.columns else gdf.columns[2]
    
    # Structural description mapping locked down explicitly
    structure_desc_col = 'lbcs_structure_desc' if 'lbcs_structure_desc' in gdf.columns else gdf.columns[3]
    unit_count_col = 'll_address_count' if 'll_address_count' in gdf.columns else gdf.columns[4]
    lot_area_col = 'gissqft' if 'gissqft' in gdf.columns else gdf.columns[5]
    
    # AI Massing model characteristics fallbacks (checks columns cleanly)
    height_col = 'height' if 'height' in gdf.columns else None
    footprint_col = 'footprint' if 'footprint' in gdf.columns else None
    coverage_col = 'coverage' if 'coverage' in gdf.columns else None
    far_col = 'far' if 'far' in gdf.columns else None
    shape_col = 'shape' if 'shape' in gdf.columns else None
    width_col = 'width' if 'width' in gdf.columns else None
    depth_col = 'depth' if 'depth' in gdf.columns else None

    # --- DYNAMIC DECADE CREATOR FROM YR_BLT ---
    if year_col and year_col in gdf.columns:
        gdf['numeric_year'] = pd.to_numeric(gdf[year_col], errors='coerce')
        gdf['decade'] = gdf['numeric_year'].apply(
            lambda x: f"{int(str(int(x))[:3]) * 10}s" if pd.notnull(x) and len(str(int(x))) == 4 else "Unknown Era"
        )
    else:
        gdf['decade'] = "Unknown Era"

    # 3. Create Sidebar Filters
    st.sidebar.header("🗺️ Filter Parcels")
    
    zoning_options = ["All"] + sorted(list(gdf[zoning_col].astype(str).unique()))
    selected_zoning = st.sidebar.selectbox("Select Zoning District", zoning_options)
    
    context_options = ["All"] + sorted(list(gdf[structure_desc_col].astype(str).unique()))
    selected_context = st.sidebar.selectbox("Select Structure Use Description", context_options)
    
    decade_options = ["All"] + sorted(list(gdf['decade'].astype(str).unique()))
    selected_decade = st.sidebar.selectbox("Select Construction Era (Decade)", decade_options)
    
    # Filter geographic data dynamically
    filtered_gdf = gdf.copy()
    if selected_zoning != "All":
        filtered_gdf = filtered_gdf[filtered_gdf[zoning_col] == selected_zoning]
    if selected_context != "All":
        filtered_gdf = filtered_gdf[filtered_gdf[structure_desc_col] == selected_context]
    if selected_decade != "All":
        filtered_gdf = filtered_gdf[filtered_gdf['decade'] == selected_decade]
        
    # 4. Layout Columns: Map on left, AI Stats and 3D Model on the right
    col1, col2 = st.columns([5, 4])
    
    with col1:
        st.subheader("Interactive Prediction
