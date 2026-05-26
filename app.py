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
        # FIXED: Line 89 is kept strictly on a single unbroken string sequence
        st.subheader("Interactive Prediction Map")
        m = folium.Map(location=[31.85, -106.41], zoom_start=12, tiles="cartodbpositron")
        
        if not filtered_gdf.empty:
            display_fields = [zoning_col, structure_desc_col, address_col]
            display_aliases = ['Zoning:', 'Structure Profile:', 'Address:']
            
            popup = folium.GeoJsonPopup(fields=display_fields, aliases=display_aliases, localize=True, labels=True)
            folium.GeoJson(
                filtered_gdf.__geo_interface__,
                style_function=lambda x: {'fillColor': '#3186cc', 'color': '#000000', 'weight': 1, 'fillOpacity': 0.6},
                popup=popup
            ).add_to(m)
            
        st_data = st_folium(m, width=700, height=450)
        
        # --- HISTORICAL TIMELINE DEVELOPMENT TREND CHARTS ---
        st.subheader("📈 Regional Density & Lot Characteristics Over Time")
        if not filtered_gdf.empty:
            gdf_numeric = filtered_gdf.copy()
            gdf_numeric['lot_size_numeric'] = pd.to_numeric(gdf_numeric[lot_area_col], errors='coerce')
            gdf_numeric['units_numeric'] = pd.to_numeric(gdf_numeric[unit_count_col], errors='coerce')
            
            timeline_df = gdf_numeric.groupby('decade').agg(
                avg_lot_size=('lot_size_numeric', 'mean'),
                avg_density=('units_numeric', 'mean')
            ).reset_index()
            
            timeline_df = timeline_df[timeline_df['decade'] != "Unknown Era"].sort_values('decade')
            timeline_df['decade'] = timeline_df['decade'].astype(str)
            
            if not timeline_df.empty:
                chart_tab1, chart_tab2 = st.tabs(["📐 Average Lot Size Trend", "👥 Unit Density Trend"])
                with chart_tab1:
                    fig_lot = px.area(timeline_df, x='decade', y='avg_lot_size', markers=True,
                                      labels={'decade': 'Construction Decade', 'avg_lot_size': 'Avg Lot Size (sqft)'},
                                      title='Evolution of Average Parcel Subdivision Scale')
                    fig_lot.update_traces(line=dict(color='#2c3e50', width=3), fillcolor='rgba(44, 62, 80, 0.15)')
                    fig_lot.update_layout(xaxis_type='category')
                    st.plotly_chart(fig_lot, use_container_width=True)
                with chart_tab2:
                    fig_density = px.bar(timeline_df, x='decade', y='avg_density',
                                         labels={'decade': 'Construction Decade', 'avg_density': 'Avg Unit Count per Lot'},
                                         title='Evolution of Multi-family Development Intensity')
                    fig_density.update_traces(marker_color='#3186cc')
                    fig_density.update_layout(xaxis_type='category')
                    st.plotly_chart(fig_density, use_container_width=True)
            else:
                st.info("Insufficient timeline history available for this specific split filter to plot charts.")
        
        # --- COMPLETE PROPERTY ARCHITECTURAL LOG TABLE ---
        st.subheader("📋 Complete Architectural & Property Log")
        table_mapping = [
            (address_col, "Property Address"),
            (structure_desc_col, "Structure Typology Description"),
            (unit_count_col, "Residential Units Count"),
            (lot_area_col, "Lot Size (sqft)"),
            (year_col, "Year Built")
        ]
        if height_col: table_mapping.append((height_col, "Predicted Height (ft)"))
        if far_col: table_mapping.append((far_col, "Floor Area Ratio (FAR)"))
        if coverage_col: table_mapping.append((coverage_col, "Lot Coverage"))
        if footprint_col: table_mapping.append((footprint_col, "Footprint Size (sqft)"))
        
        if not filtered_gdf.empty:
            display_df = pd.DataFrame()
            for db_col, clean_title in table_mapping:
                if db_col in filtered_gdf.columns:
                    display_df[clean_title] = filtered_gdf[db_col]
            
            if "Lot Size (sqft)" in display_df.columns:
                display_df["Lot Size (sqft)"] = pd.to_numeric(display_df["Lot Size (sqft)"], errors='coerce').apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "0")
            if "Property Address" in display_df.columns:
                display_df["Property Address"] = display_df["Property Address"].astype(str).str.upper()
            if "Structure Typology Description" in display_df.columns:
                display_df["Structure Typology Description"] = display_df["Structure Typology Description"].astype(str).str.title()
                
            st.dataframe(display_df.head(50), use_container_width=True)
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

        # Baseline envelope dimensional characteristics
        avg_parcel_area = get_safe_mean(active_source, lot_area_col, 12000.0)
        avg_units = get_safe_mean(active_source, unit_count_col, 8.0)
        
        avg_w = get_safe_mean(active_source, width_col, 80.0)
        avg_d = get_safe_mean(active_source, depth_col, 50.0)
        
        # Pull or calculate structural envelopes 
        avg_footprint = get_safe_mean(active_source, footprint_col, avg_parcel_area * 0.38)
        avg_height = get_safe_mean(active_source, height_col, 24.0 if avg_units <= 8 else 45.0)
        avg_coverage = get_safe_mean(active_source, coverage_col, min(avg_footprint / max(avg_parcel_area, 1.0), 0.65))
        
        # Custom dynamic architectural calculation loop for FAR
        estimated_stories = max(avg_height / 12.0, 1.0)
        total_building_area = avg_footprint * estimated_stories
        
        avg_far = float(total_building_area / max(avg_parcel_area, 1.0))
        avg_far = float(np.clip(avg_far, 0.05, 12.0))

        if structure_desc_col in active_source.columns and not active_source[structure_desc_col].dropna().empty:
            most_common_desc = str(active_source[structure_desc_col].dropna().mode()[0]).title()
        else:
            most_common_desc = "Multifamily Apartment"

        if avg_w / max(avg_d, 1.0) > 1.8:
            inferred_shape = "Elongated Linear Bar"
        elif avg_units > 24:
            inferred_shape = "Courtyard Block / Complex"
        else:
            inferred_shape = "Compact Rectangle / Box"
            
        # Render Metrics Card Matrix Array
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric("Parcels In Focus", f"{len(filtered_gdf):,}")
            st.metric("Avg Footprint Square Footage", f"{avg_footprint:,.0f} sq ft")
            st.metric("Avg Estimated Height", f"{avg_height:.1f} ft")
        with m_col2:
            st.metric("Avg Base Lot Size", f"{avg_parcel_area:,.0f} sq ft")
            st.metric("Avg Lot Coverage Factor", f"{avg_coverage * 100:.1f}%" if avg_coverage <= 1.0 else f"{avg_coverage:.1f}%")
            st.metric("Avg Floor Area Ratio (FAR)", f"{avg_far:.2f}")
            
        # --- EXPLICIT TEXT SHAPE CALLOUT BANNER ---
        st.markdown("---")
        st.info(f"🔮 **AI Portfolio Design Typology:** The models parse your selected criteria as a **{most_common_desc}**. Based on development constraints, the optimal footprint massing fits a **{inferred_shape}** layout configuration.")
        
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
