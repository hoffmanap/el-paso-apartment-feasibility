import os
import streamlit as st
import geopandas as gpd
import folium
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_folium import st_folium

# 1. Set up the web page configurations
st.set_page_config(page_title="El Paso Apartment Feasibility AI", layout="wide")

st.title("🏢 El Paso Apartment Design & Feasibility AI")
st.markdown("Explore existing multifamily parcels and look at predicted AI massing variables based on zoning, context, and lot features.")

# 2. Path to your prediction outputs
data_dir = r"C:\Users\Angelica\OneDrive\Apartment Analysis\data"
prediction_file = os.path.join(data_dir, "multifamily_predictions.geojson")

@st.cache_data
def load_web_data(path):
    if not os.path.exists(path):
        return None
    gdf = gpd.read_file(path)
    
    # 100% BULLETPROOF COLUMN CLEANING: Flatten headers and strip hidden spatial nesting
    cleaned_cols = []
    for c in gdf.columns:
        c_clean = str(c).replace('properties/', '').lower().strip()
        cleaned_cols.append(c_clean)
    gdf.columns = cleaned_cols
    
    return gdf.to_crs(epsg=4326)

gdf = load_web_data(prediction_file)

if gdf is None:
    st.error(f"Could not find the predictions file at {prediction_file}. Please make sure you've successfully run predict_footprints.py first!")
else:
    # --- DYNAMIC COLUMN HELPER (Finds columns even if spelling shifts) ---
    def find_col(possible_names, contains_keywords):
        for name in possible_names:
            if name in gdf.columns:
                return name
        for keyword in contains_keywords:
            for col in gdf.columns:
                if keyword in col:
                    return col
        return None

    # Resolve all core database columns dynamically
    zoning_col = find_col(['zoning', 'zoning_district'], ['zone']) or gdf.columns[0]
    context_col = find_col(['context', 'geographic_context'], ['context', 'geo']) or gdf.columns[1]
    address_col = find_col(['prop_add', 'address', 'prop_addr'], ['add', 'location']) or gdf.columns[2]
    
    height_col = find_col(['height', 'pred_height'], ['height', 'ft'])
    footprint_col = find_col(['footprint', 'pred_footprint'], ['foot', 'print', 'sq'])
    coverage_col = find_col(['coverage', 'pred_coverage'], ['cover', 'pct'])
    far_col = find_col(['far', 'pred_far'], ['far', 'ratio'])
    shape_col = find_col(['shape', 'pred_shape'], ['shape', 'type', 'mass'])
    width_col = find_col(['width', 'pred_width'], ['width', 'dim_x'])
    depth_col = find_col(['depth', 'pred_depth'], ['depth', 'dim_y'])
    
    # Lot area column fallback tracking
    lot_area_col = find_col(['gissqft', 'll_gissqft', 'parcel_area'], ['sqft', 'lot', 'area']) or gdf.columns[3]
    
    # Year/Decade extraction column locator
    year_col = find_col(['yr_built', 'year_built', 'year', 'yrbuilt'], ['year', 'built', 'date', 'yr'])

    # --- DYNAMIC DECADE CREATOR ---
    if year_col:
        # Convert to numbers safely and drop empty values for calculation
        gdf['numeric_year'] = pd.to_numeric(gdf[year_col], errors='coerce')
        
        # Calculate the decade tag (e.g., 1984 -> "1980s")
        gdf['decade'] = gdf['numeric_year'].apply(
            lambda x: f"{int(str(x)[:3]) * 10}s" if pd.notnull(x) and len(str(int(x))) == 4 else "Unknown Era"
        )
    else:
        gdf['decade'] = "Unknown Era"

    # 3. Create Sidebar Filters
    st.sidebar.header("🗺️ Filter Parcels")
    
    zoning_options = ["All"] + sorted(list(gdf[zoning_col].astype(str).unique()))
    selected_zoning = st.sidebar.selectbox("Select Zoning District", zoning_options)
    
    context_options = ["All"] + sorted(list(gdf[context_col].astype(str).unique()))
    selected_context = st.sidebar.selectbox("Select Geographic Context", context_options)
    
    # NEW: Decade Filter added dynamically to the side panel layout
    decade_options = ["All"] + sorted(list(gdf['decade'].unique()))
    selected_decade = st.sidebar.selectbox("Select Construction Era (Decade)", decade_options)
    
    # Apply user selections to filter the geographic data dynamically
    filtered_gdf = gdf.copy()
    if selected_zoning != "All":
        filtered_gdf = filtered_gdf[filtered_gdf[zoning_col] == selected_zoning]
    if selected_context != "All":
        filtered_gdf = filtered_gdf[filtered_gdf[context_col] == selected_context]
    if selected_decade != "All":
        filtered_gdf = filtered_gdf[filtered_gdf['decade'] == selected_decade]
        
    # 4. Layout Columns: Map on left, AI Stats and 3D Model on the right
    col1, col2 = st.columns([5, 4])
    
    with col1:
        st.subheader("Interactive Prediction Map")
        m = folium.Map(location=[31.85, -106.41], zoom_start=12, tiles="cartodbpositron")
        
        if not filtered_gdf.empty:
            display_fields = [zoning_col, context_col]
            display_aliases = ['Zoning:', 'Context:']
            
            for col_var, alias_text in [(shape_col, 'Shape:'), (height_col, 'Height (ft):'), (footprint_col, 'Footprint (sqft):'), (coverage_col, 'Coverage (%):'), (far_col, 'FAR:')]:
                if col_var and col_var in filtered_gdf.columns:
                    display_fields.append(col_var)
                    display_aliases.append(alias_text)
            
            popup = folium.GeoJsonPopup(fields=display_fields, aliases=display_aliases, localize=True, labels=True)
            folium.GeoJson(
                filtered_gdf.__geo_interface__,
                style_function=lambda x: {'fillColor': '#3186cc', 'color': '#000000', 'weight': 1, 'fillOpacity': 0.6},
                popup=popup
            ).add_to(m)
            
        st_data = st_folium(m, width=700, height=550)
        
        # --- FIXED & RE-MAPPED ARCHITECTURAL LOG TABLE ---
        st.subheader("📋 Complete Architectural & Property Log")
        
        # Build out table variables based on requested layout specifications
        table_mapping = []
        if address_col and address_col in gdf.columns: table_mapping.append((address_col, "Property Address"))
        if year_col and year_col in gdf.columns: table_mapping.append((year_col, "Year Built"))
        if shape_col and shape_col in gdf.columns: table_mapping.append((shape_col, "Shape Typology"))
        if height_col and height_col in gdf.columns: table_mapping.append((height_col, "Predicted Height (ft)"))
        if far_col and far_col in gdf.columns: table_mapping.append((far_col, "Floor Area Ratio (FAR)"))
        if coverage_col and coverage_col in gdf.columns: table_mapping.append((coverage_col, "Lot Coverage"))
        if lot_area_col and lot_area_col in gdf.columns: table_mapping.append((lot_area_col, "Lot Size (sqft)"))
        if footprint_col and footprint_col in gdf.columns: table_mapping.append((footprint_col, "Footprint Size (sqft)"))
        
        active_table_cols = [item[0] for item in table_mapping]
        rename_dict = {item[0]: item[1] for item in table_mapping}
        
        if not filtered_gdf.empty:
            # Force inclusion of columns even if they didn't match perfectly, by using our safe search array
            display_df = pd.DataFrame()
            for db_col, clean_title in table_mapping:
                if db_col in filtered_gdf.columns:
                    display_df[clean_title] = filtered_gdf[db_col]
            
            # Numeric cleanups for clean formatting inside the table cells safely
            if "Predicted Height (ft)" in display_df.columns:
                display_df["Predicted Height (ft)"] = pd.to_numeric(display_df["Predicted Height (ft)"], errors='coerce').round(1)
            if "Floor Area Ratio (FAR)" in display_df.columns:
                display_df["Floor Area Ratio (FAR)"] = pd.to_numeric(display_df["Floor Area Ratio (FAR)"], errors='coerce').round(2)
            if "Lot Size (sqft)" in display_df.columns:
                display_df["Lot Size (sqft)"] = pd.to_numeric(display_df["Lot Size (sqft)"], errors='coerce').apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "0")
            if "Footprint Size (sqft)" in display_df.columns:
                display_df["Footprint Size (sqft)"] = pd.to_numeric(display_df["Footprint Size (sqft)"], errors='coerce').apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "0")
            if "Shape Typology" in display_df.columns:
                display_df["Shape Typology"] = display_df["Shape Typology"].astype(str).str.title()
            if "Property Address" in display_df.columns:
                display_df["Property Address"] = display_df["Property Address"].astype(str).str.upper()
                
            # Format coverage beautifully
            if "Lot Coverage" in display_df.columns:
                display_df["Lot Coverage"] = pd.to_numeric(display_df["Lot Coverage"], errors='coerce').apply(
                    lambda x: f"{x*100:.1f}%" if pd.notnull(x) and float(x) <= 1.0 else (f"{x:.1f}%" if pd.notnull(x) else "0.0%")
                )
                
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No rows to display for this filter combination.")
        
    with col2:
        st.subheader("📊 Selected Portfolio Insights")
        
        active_source = filtered_gdf if not filtered_gdf.empty else gdf
        if filtered_gdf.empty:
            st.warning("⚠️ No matching rows found for this selection. Showing global city summary data instead:")
            
        # Robust numeric calculation function
        def get_safe_mean(df_source, target_column, default_value):
            if target_column and target_column in df_source.columns:
                series = pd.to_numeric(df_source[target_column], errors='coerce').dropna()
                if not series.empty:
                    val = series.mean()
                    return float(val) if pd.notnull(val) else default_value
            return default_value

        avg_height = get_safe_mean(active_source, height_col, 35.0)
        avg_footprint = get_safe_mean(active_source, footprint_col, 4500.0)
        avg_coverage = get_safe_mean(active_source, coverage_col, 0.40)
        avg_far = get_safe_mean(active_source, far_col, 0.85)
        avg_parcel_area = get_safe_mean(active_source, lot_area_col, 12000.0)
        
        avg_w = get_safe_mean(active_source, width_col, 80.0)
        avg_d = get_safe_mean(active_source, depth_col, 50.0)
        
        if shape_col and shape_col in active_source.columns and not active_source[shape_col].dropna().empty:
            most_common_shape = str(active_source[shape_col].dropna().mode()[0]).title()
        else:
            most_common_shape = "Rectangle / Box"
            
        # Render Metrics
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric("Parcels In Focus", f"{len(filtered_gdf):,}")
            st.metric("Avg Footprint Square Footage", f"{avg_footprint:,.0f} sq ft")
            st.metric("Avg Predicted Height", f"{avg_height:.1f} ft")
        with m_col2:
            st.metric("Avg Base Lot Size", f"{avg_parcel_area:,.0f} sq ft")
            
            if avg_coverage > 0.0 and avg_coverage <= 1.0:
                st.metric("Avg Lot Coverage Factor", f"{avg_coverage * 100:.1f}%")
            else:
                st.metric("Avg Lot Coverage Factor", f"{avg_coverage:.1f}%")
                
            st.metric("Avg Floor Area Ratio (FAR)", f"{avg_far:.2f}")
            
        # --- EXPLICIT TEXT SHAPE CALLOUT BANNER ---
        st.markdown("---")
        st.info(f"🔮 **AI Predicted Building Configuration:** The models indicate that the most structurally optimal design typology for this selection is a **{most_common_shape}** layout configuration.")
        
        # Render 3D Box Model
        st.subheader("📦 3D Architectural Envelope Massing")
        
        color_map = {
            'rectangle / box': 'rgba(49, 134, 204, 0.8)',
            'l-shape / t-shape': 'rgba(230, 126, 34, 0.8)',
            'complex / courtyard': 'rgba(155, 89, 182, 0.8)'
        }
        mesh_color = color_map.get(most_common_shape.lower().strip(), 'rgba(49, 134, 204, 0.8)')
        
        w, d, h = max(avg_w, 20.0), max(avg_d, 20.0), max(avg_height, 10.0)
        
        x_coords = [0, w, w, 0, 0, w, w, 0]
        y_coords = [0, 0, d, d, 0, 0, d, d]
        z_coords = [0, 0, 0, 0, h, h, h, h]
        
        fig = go.Figure(data=[
            go.Mesh3d(
                x=x_coords, y=y_coords, z=z_coords,
                i=[7, 0, 0, 0, 4, 4, 1, 2, 2, 3, 4, 5],
                j=[0, 4, 1, 2, 5, 6, 2, 3, 7, 7, 7, 6],
                k=[3, 7, 5, 6, 1, 2, 6, 7, 6, 0, 5, 1],
                opacity=0.85,
                color=mesh_color,
                flatshading=True
            )
        ])
        
        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(
                xaxis=dict(title='Width (ft)', range=[-10, max(w+20, 150)]),
                yaxis=dict(title='Depth (ft)', range=[-10, max(d+20, 150)]),
                zaxis=dict(title='Height (ft)', range=[0, max(h+20, 60)]),
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.5)
            ),
            height=320,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("✨ Note: Drag your mouse over the box above to rotate, inspect, and pan around the predicted 3D envelope structure!")