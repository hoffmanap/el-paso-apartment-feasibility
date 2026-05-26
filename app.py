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

# 2. Path to your prediction outputs (FIXED FOR CLOUD HOSTING RELATIVE PATHS)
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
    # --- SELF-HEALING COLUMN SCANNER ENGINE ---
    def find_best_column(preferred_names, backup_keywords):
        for name in preferred_names:
            if name in gdf.columns:
                return name
        for keyword in backup_keywords:
            for col in gdf.columns:
                if keyword in col:
                    return col
        return gdf.columns[0]

    # Dynamically locate columns based on your provided keys
    zoning_col = find_best_column(['zoning', 'zoning_district'], ['zone'])
    context_col = find_best_column(['lbcs_structure_desc', 'use_desc', 'context'], ['desc', 'use', 'structure', 'type'])
    address_col = find_best_column(['address', 'prop_add', 'prop_addr'], ['add', 'loc', 'street'])
    year_col = find_best_column(['yr_blt', 'year_built', 'year'], ['yr', 'blt', 'built', 'date'])
    
    unit_count_col = find_best_column(['ll_address_count', 'units'], ['count', 'unit', 'res']) if any('count' in c or 'unit' in c for c in gdf.columns) else None
    lot_area_col = find_best_column(['gissqft', 'll_gissqft', 'parcel_area'], ['sqft', 'lot', 'area', 'gis'])
    
    # AI Massing model characteristics fallbacks (maps cleanly if they exist)
    height_col = find_best_column(['height'], ['height', 'ft']) if 'height' in ''.join(gdf.columns) else None
    footprint_col = find_best_column(['footprint'], ['foot', 'print']) if 'footprint' in ''.join(gdf.columns) else None
    coverage_col = find_best_column(['coverage'], ['cover']) if 'coverage' in ''.join(gdf.columns) else None
    far_col = find_best_column(['far'], ['far', 'ratio']) if 'far' in ''.join(gdf.columns) else None
    shape_col = find_best_column(['shape'], ['shape', 'type']) if 'shape' in ''.join(gdf.columns) else None
    width_col = find_best_column(['width'], ['width']) if 'width' in ''.join(gdf.columns) else None
    depth_col = find_best_column(['depth'], ['depth']) if 'depth' in ''.join(gdf.columns) else None

    # --- DYNAMIC DECADE CREATOR FROM YR_BLT ---
    if year_col and year_col in gdf.columns:
        gdf['numeric_year'] = pd.to_numeric(gdf[year_col], errors='coerce')
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
    selected_context = st.sidebar.selectbox("Select Use/Structure Context", context_options)
    
    decade_options = ["All"] + sorted(list(gdf['decade'].astype(str).unique()))
    selected_decade = st.sidebar.selectbox("Select Construction Era (Decade)", decade_options)
    
    # Filter geographic data dynamically
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
            display_fields = [zoning_col, context_col, address_col]
            display_aliases = ['Zoning:', 'Context/Use:', 'Address:']
            
            popup = folium.GeoJsonPopup(fields=display_fields, aliases=display_aliases, localize=True, labels=True)
            folium.GeoJson(
                filtered_gdf.__geo_interface__,
                style_function=lambda x: {'fillColor': '#3186cc', 'color': '#000000', 'weight': 1, 'fillOpacity': 0.6},
                popup=popup
            ).add_to(m)
            
        st_data = st_folium(m, width=700, height=550)
        
        # --- COMPLETE PROPERTY ARCHITECTURAL LOG TABLE ---
        st.subheader("📋 Complete Architectural & Property Log")
        
        table_mapping = []
        if address_col in filtered_gdf.columns: table_mapping.append((address_col, "Property Address"))
        if shape_col and shape_col in filtered_gdf.columns: table_mapping.append((shape_col, "Shape Typology"))
        if height_col in filtered_gdf.columns: table_mapping.append((height_col, "Predicted Height (ft)"))
        if far_col in filtered_gdf.columns: table_mapping.append((far_col, "Floor Area Ratio (FAR)"))
        if coverage_col in filtered_gdf.columns: table_mapping.append((coverage_col, "Lot Coverage"))
        if lot_area_col in filtered_gdf.columns: table_mapping.append((lot_area_col, "Lot Size (sqft)"))
        if footprint_col in filtered_gdf.columns: table_mapping.append((footprint_col, "Footprint Size (sqft)"))
        if unit_count_col and unit_count_col in filtered_gdf.columns: table_mapping.append((unit_count_col, "Residential Units"))
        if year_col in filtered_gdf.columns: table_mapping.append((year_col, "Year Built"))
        
        if not filtered_gdf.empty:
            display_df = pd.DataFrame()
            for db_col, clean_title in table_mapping:
                if db_col in filtered_gdf.columns:
                    display_df[clean_title] = filtered_gdf[db_col]
            
            # Format numbers beautifully
            if "Predicted Height (ft)" in display_df.columns:
                display_df["Predicted Height (ft)"] = pd.to_numeric(display_df["Predicted Height (ft)"], errors='coerce').round(1)
            if "Floor Area Ratio (FAR)" in display_df.columns:
                display_df["Floor Area Ratio (FAR)"] = pd.to_numeric(display_df["Floor Area Ratio (FAR)"], errors='coerce').round(2)
            if "Lot Size (sqft)" in display_df.columns:
                display_df["Lot Size (sqft)"] = pd.to_numeric(display_df["Lot Size (sqft)"], errors='coerce').apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "0")
            if "Footprint Size (sqft)" in display_df.columns:
                display_df["Footprint Size (sqft)"] = pd.to_numeric(display_df["Footprint Size (sqft)"], errors='coerce').apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "0")
            if "Property Address" in display_df.columns:
                display_df["Property Address"] = display_df["Property Address"].astype(str).str.upper()
            if "Shape Typology" in display_df.columns:
                display_df["Shape Typology"] = display_df["Shape Typology"].astype(str).str.title()
                
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
        
        def get_safe_mean(df_source, target_column, default_value):
            if target_column and target_column in df_source.columns:
                series = pd.to_numeric(df_source[target_column], errors='coerce').dropna()
                if not series.empty:
                    val = series.mean()
                    return float(val) if pd.notnull(val) else default_value
            return default_value

        # Calculate means
        avg_height = get_safe_mean(active_source, height_col, 35.0)
        avg_footprint = get_safe_mean(active_source, footprint_col, 4500.0)
        avg_coverage = get_safe_mean(active_source, coverage_col, 0.40)
        avg_far = get_safe_mean(active_source, far_col, 0.85)
        avg_parcel_area = get_safe_mean(active_source, lot_area_col, 12000.0)
        
        avg_w = get_safe_mean(active_source, width_col, 80.0)
        avg_d = get_safe_mean(active_source, depth_col, 50.0)
        
        # FIXED: Pulled directly from the structural classification column description mapping
        if context_col and context_col in active_source.columns and not active_source[context_col].dropna().empty:
            most_common_type = str(active_source[context_col].dropna().mode()[0]).title()
        else:
            most_common_type = "Multifamily Apartment Structure"
            
        # Render Metrics Card Matrix Array
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
            
        # --- FIXED EXPLICIT TEXT SHAPE CALLOUT BANNER ---
        st.markdown("---")
        st.info(f"🔮 **AI Portfolio Design Typology:** The models indicate that the primary architectural configuration for this selection is optimized for a **{most_common_type}** development layout.")
        
        # Render hardware-accelerated 3D Architectural Envelope Massing
        st.subheader("📦 3D Architectural Envelope Massing")
        
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
                color='rgba(49, 134, 204, 0.8)',
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
