"""
Project: European Electricity Exchange Analysis
Author: Tiernan Buckley
Year: 2026
License: Creative Commons Attribution 4.0 International (CC BY 4.0)
Source: https://github.com/INATECH-CIG/exchange_analysis

Description:
Interactive Streamlit dashboard for visualizing the European Electricity Market 
Exchange Analysis. This module renders dynamic geographic flow maps, tracks 
bidding zone net positions, and plots high-resolution hourly internal generation 
and imported fuel mixes using Plotly and GeoPandas.
"""

import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
import json
import numpy as np
from pathlib import Path
from entsoe.geo.utils import load_zones
from mappings_alt import NEIGHBOURS
from utils import io

# ==========================================
# 0. CONFIG & CONSTANTS
# ==========================================

# Defines file paths, database tables, and logic types for each methodology
FLOW_TYPES = {
    "Physical": {
        "flow_path": "physical_flow_data_bidding_zones/{year}/{bz}_physical_flow_data_bidding_zones.csv",
        "mix_path": None, 
        "flow_table": "processed_physical_flows",
        "mix_table": None,
        "type": "standard"
    },
    "Commercial Total": {
        "flow_path": "comm_flow_total_bidding_zones/{year}/{bz}_comm_flow_total_bidding_zones.csv",
        "mix_path": "comm_flow_total_bidding_zones/{year}/results/netted_per_type/{bz}_import_comm_flow_total_netted_per_type.csv",
        "flow_table": "processed_commercial_flows",
        "mix_table": "analysis_cft_netted_type",
        "type": "standard"
    },
    "Commercial Day-Ahead": {
        "flow_path": "comm_flow_dayahead_bidding_zones/{year}/{bz}_comm_flow_dayahead_bidding_zones.csv",
        "mix_path": None,
        "flow_table": "processed_commercial_flows_da",
        "mix_table": None,
        "type": "standard"
    },
    "Agg. Coupling Flow Tracing": {
        "flow_path": "import_flow_tracing_bidding_zones/agg_coupling/{year}/per_bidding_zone/{bz}_import_flow_tracing_agg_coupling_per_bidding_zone.csv",
        "mix_path": "import_flow_tracing_bidding_zones/agg_coupling/{year}/per_type/{bz}_import_flow_tracing_agg_coupling_per_type.csv",
        "flow_table": "tracing_agg_coupling_bz",
        "mix_table": "tracing_agg_coupling_type",
        "type": "tracing"
    },
    "Direct Coupling Flow Tracing": {
        "flow_path": "import_flow_tracing_bidding_zones/direct_coupling/{year}/per_bidding_zone/{bz}_import_flow_tracing_direct_coupling_per_bidding_zone.csv",
        "mix_path": "import_flow_tracing_bidding_zones/direct_coupling/{year}/per_type/{bz}_import_flow_tracing_direct_coupling_per_type.csv",
        "flow_table": "tracing_direct_coupling_bz",
        "mix_table": "tracing_direct_coupling_type",
        "type": "tracing"
    },
    "Net Pooled CFT": {
        "flow_path": "pooling/{year}/commercial_net_pos/per_bidding_zone/{bz}_pooled_net_imports_per_bidding_zone.csv",
        "mix_path": "pooling/{year}/commercial_net_pos/per_type/{bz}_pooled_net_imports_per_type.csv",
        "flow_table": "pool_commercial_net_pos_bz",
        "mix_table": "pool_commercial_net_pos_type",
        "type": "tracing"
    }
}

# Standardized color mapping for electricity generation sources
GEN_COLORS = {
    "Solar": "#f1c40f", "Wind Onshore": "#3498db", "Wind Offshore": "#2980b9",
    "Biomass": "#27ae60", "Hydro Water Reservoir": "#1abc9c", 
    "Hydro Run-of-river and poundage": "#16a085", "Geothermal": "#d35400",
    "Marine": "#16a085", "Other renewable": "#2ecc71", "Nuclear": "#9b59b6",
    "Fossil Gas": "#e67e22", "Fossil Hard coal": "#34495e", "Fossil Oil": "#2c3e50",
    "Waste": "#7f8c8d", "Fossil Brown coal/Lignite": "#795548", 
    "Fossil Coal-derived gas": "#5d4037", "Fossil Oil shale": "#4e342e",
    "Fossil Peat": "#3e2723", "Storage Discharge": "#546e7a",
    "Storage Charge": "#78909c", "Storage": "#546e7a", "Other": "#bdc3c7"
}

# Handles environment-specific pathing and time formatting
class MockConfig:
    def __init__(self, selected_date):
        self.year = selected_date.year
        self.output_dir = Path(__file__).parent.parent / "outputs"
        self.load_source, self.save_db, self.save_csv = 'db', True, True
        self.start = pd.Timestamp(selected_date).tz_localize("UTC")
        self.end = self.start + pd.Timedelta(hours=23, minutes=59, seconds=59)

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================

def get_clean_zones():
    """Returns bidding zones used in analysis, excluding overlapping sub-aggregations."""
    to_remove = ["DE_AT_LU", "IE_SEM", "IE", "NIE", "MT", "IT", "IT_BRNN", "IT_ROSN", "IT_FOGN"]
    return sorted([z for z in NEIGHBOURS.keys() if z not in to_remove])

def get_bearing(lon1, lat1, lon2, lat2):
    """Calculates rotation angle for arrowhead markers on the map."""
    d_lon = lon2 - lon1
    y = np.sin(np.radians(d_lon)) * np.cos(np.radians(lat2))
    x = np.cos(np.radians(lat1)) * np.sin(np.radians(lat2)) - \
        np.sin(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.cos(np.radians(d_lon))
    return (np.degrees(np.arctan2(y, x)) + 360) % 360

def get_curve(p1, p2, num_points=20):
    """Generates arc points to visually separate import and export lines."""
    lons, lats = np.linspace(p1[0], p2[0], num_points), np.linspace(p1[1], p2[1], num_points)
    dist = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
    offset = dist * 0.15 
    shift = offset * np.sin(np.linspace(0, np.pi, num_points))
    return lons + shift*0.2, lats + shift

# ==========================================
# 2. DATA LOADING (Cached)
# ==========================================

@st.cache_data
def load_geography(active_zones):
    """Loads and merges ENTSO-E zone geometries with local custom GeoJSONs."""
    custom_zones = ["GB", "ME", "BA", "MK"]
    entsoe_zones = [z for z in active_zones if z not in custom_zones]
    geo_df = load_zones(entsoe_zones, pd.Timestamp('2024-01-01'))
    input_dir = Path(__file__).parent.parent / "inputs"
    for country in custom_zones:
        try:
            zone = gpd.read_file(input_dir / f"zones/{country}.geojson")
            geo_df.loc[country] = zone["geometry"][0]
        except: continue
    geo_df['lon'], geo_df['lat'] = geo_df.geometry.centroid.x, geo_df.geometry.centroid.y
    return geo_df.drop(["geometry"], axis=1), json.loads(geo_df.to_json())

def _resolve_path(mock_config, template, bz):
    """Utility to handle nested year folders vs. flat directory structures."""
    ideal_path = mock_config.output_dir / template.format(year=mock_config.year, bz=bz)
    if not ideal_path.exists():
        fallback = mock_config.output_dir / template.format(year=mock_config.year, bz=bz).replace(f"/{mock_config.year}/", "/")
        if fallback.exists(): return fallback
    return ideal_path

@st.cache_data
def load_full_day_data(selected_date, target_bz, flow_settings):
    """Loads the core exchange matrix for the selected methodology."""
    mock_config = MockConfig(selected_date)
    path = _resolve_path(mock_config, flow_settings["flow_path"], target_bz)
    return io.load(path, flow_settings["flow_table"], mock_config, bz=target_bz)

@st.cache_data
def load_generation_data(selected_date, target_bz):
    """Loads internal generation and demand data for the specific zone."""
    mock_config = MockConfig(selected_date)
    template = "generation_demand_data_bidding_zones/{year}/{bz}_generation_demand_data_bidding_zones.csv"
    path = _resolve_path(mock_config, template, target_bz)
    return io.load(path, "processed_generation", mock_config, bz=target_bz)

@st.cache_data
def load_import_mix(selected_date, target_bz, flow_settings):
    """Loads the fuel-type breakdown of imported energy (if calculated)."""
    template = flow_settings.get("mix_path")
    table_name = flow_settings.get("mix_table")
    if not template or not table_name: return None
    mock_config = MockConfig(selected_date)
    path = _resolve_path(mock_config, template, target_bz)
    df = io.load(path, table_name, mock_config, bz=target_bz)
    if df is not None and not df.empty:
        rename_map = {col: col.split("_")[-1] for col in df.columns if "_" in col}
        df = df.rename(columns=rename_map)
        return df.T.groupby(level=0).sum().T
    return df

def extract_arrow_flows(target_bz, hourly_all, active_zones, flow_settings, selected_date, target_time):
    """Iterates through all possible neighbors to extract directional MW flow pairs."""
    active_flows = []
    if hourly_all is None or hourly_all.empty: return active_flows
    
    # Tracing logic requires cross-referencing neighbor imports to find exports
    if flow_settings.get("type") == "tracing":
        for source_bz in active_zones:
            if source_bz == target_bz: continue
            if source_bz in hourly_all.columns:
                val = hourly_all[source_bz].iloc[0]
                if val > 0: active_flows.append({"Source": source_bz, "Target": target_bz, "MW": val, "Type": "Import"})
        for other_bz in active_zones:
            if other_bz == target_bz: continue
            other_df = load_full_day_data(selected_date, other_bz, flow_settings)
            if other_df is not None and target_time in other_df.index:
                if target_bz in other_df.columns:
                    val = other_df.loc[target_time, target_bz]
                    if isinstance(val, pd.Series): val = val.iloc[0]
                    if val > 0: active_flows.append({"Source": target_bz, "Target": other_bz, "MW": val, "Type": "Export"})
    # Standard logic uses pre-calculated net_export columns
    else:
        for col in hourly_all.columns:
            if "_net_export" in col:
                val = hourly_all[col].iloc[0]
                if abs(val) > 0:
                    pair = col.replace("_net_export", "")
                    if pair.startswith(f"{target_bz}_"):
                        other = pair.replace(f"{target_bz}_", ""); direction = 'Import' if val < 0 else 'Export'
                        s, t = (other, target_bz) if val < 0 else (target_bz, other)
                    elif pair.endswith(f"_{target_bz}"):
                        other = pair.replace(f"_{target_bz}", ""); direction = 'Export' if val < 0 else 'Import'
                        s, t = (target_bz, other) if val < 0 else (other, target_bz)
                    active_flows.append({"Source": s, "Target": t, "MW": abs(val), "Type": direction})
    return active_flows

# ==========================================
# 3. MAP GENERATOR
# ==========================================

def draw_flow_map(geo_df, geoj, flows, target_bz, net_position_gw):
    """Renders the Plotly Choropleth + Scattergeo curved flow lines."""
    fig = go.Figure()
    hover_labels, z_values, b_colors, b_widths = [], [], [], []
    relevant_lons, relevant_lats = [geo_df.loc[target_bz, 'lon']], [geo_df.loc[target_bz, 'lat']]
    
    for zone in geo_df.index:
        if zone == target_bz:
            b_colors.append('#000000'); b_widths.append(2.0); z_values.append(1 if net_position_gw >= 0 else -1)
            hover_labels.append(f"<b>{zone}</b><br>{'Exporting' if net_position_gw >= 0 else 'Importing'}: {abs(net_position_gw):.2f} GW")
        else:
            b_colors.append('#adb5bd'); b_widths.append(0.8); z_values.append(0)
            hover_labels.append(f"<b>{zone}</b>")
            
    fig.add_trace(go.Choropleth(
        geojson=geoj, locations=geo_df.index, z=z_values, text=hover_labels, hoverinfo="text",
        colorscale=[[0, '#e3f2fd'], [0.5, '#ffffff'], [1, '#e8f5e9']], zmin=-1, zmax=1, showscale=False, 
        marker_line_color=b_colors, marker_line_width=b_widths
    ))
    
    COLOR_MAP = {'Export': {'l': 'rgba(40, 167, 69, 0.4)', 'a': '#28a745', 't': '#1e7e34'},
                 'Import': {'l': 'rgba(0, 123, 255, 0.4)', 'a': '#007bff', 't': '#0056b3'}}
                 
    for flow in flows:
        if flow['Source'] not in geo_df.index or flow['Target'] not in geo_df.index: continue
        p1, p2 = (geo_df.loc[flow['Source'], 'lon'], geo_df.loc[flow['Source'], 'lat']), (geo_df.loc[flow['Target'], 'lon'], geo_df.loc[flow['Target'], 'lat'])
        relevant_lons.extend([p1[0], p2[0]]); relevant_lats.extend([p1[1], p2[1]])
        c = COLOR_MAP.get(flow['Type'])
        cLons, cLats = get_curve(p1, p2)
        fig.add_trace(go.Scattergeo(lon=cLons, lat=cLats, mode='lines', line=dict(width=max(1.5, flow['MW']/500), color=c['l']), hoverinfo='none'))
        mid = len(cLons)//2
        fig.add_trace(go.Scattergeo(lon=[cLons[mid]], lat=[cLats[mid]], mode='markers+text', text=[f"<b>{flow['MW']/1000:.1f}</b>"], 
                                    textposition="top center", marker=dict(size=12, symbol='triangle-up', color=c['a'], angle=get_bearing(p1[0], p1[1], p2[0], p2[1]), line=dict(color='white', width=1)),
                                    textfont=dict(size=12, color=c['t'], family="Arial Black"), hoverinfo='none'))
                                    
    fig.update_layout(
        geo=dict(projection_type="mercator", lonaxis_range=[min(relevant_lons)-2.5, max(relevant_lons)+2.5], lataxis_range=[min(relevant_lats)-2.5, max(relevant_lats)+2.5], visible=False), 
        margin={"r":0,"t":0,"l":0,"b":0}, height=650, showlegend=False, clickmode='event+select', hoverlabel=dict(bgcolor="white", font_size=13, font_family="Arial")
    )
    return fig

# ==========================================
# 4. INTERFACE SETUP
# ==========================================

st.set_page_config(page_title="European Grid Analysis", layout="wide")

# Inject Custom CSS for sticky title and metric styling
st.markdown("""
    <style>
        .main > div {padding-left: 2rem; padding-right: 2rem; max-width: 100%;}
        [data-testid="stHeader"] { background-color: rgba(255, 255, 255, 0); }
        .sticky-title {
            position: fixed; top: 0; left: 0; width: 100%; background-color: white;
            padding: 1rem 2rem; margin-left: 18rem; z-index: 999; border-bottom: 1px solid #e6e6e6;
        }
        .main .block-container { padding-top: 5rem; }
        [data-testid="stMetricDelta"] svg { display: none !important; }
    </style>
    <div class="sticky-title">
        <h2 style="margin:0;">European Electricity Market Exchange Analysis</h2>
    </div>
""", unsafe_allow_html=True)

# Application state persistence
active_zones = get_clean_zones()
if "target_bz" not in st.session_state: st.session_state.target_bz = "DE_LU"
if "hour_val" not in st.session_state: st.session_state.hour_val = 12
if "flow_method" not in st.session_state: st.session_state.flow_method = "Physical"

# Sidebar controls
st.sidebar.header("Select Data:")
selected_bz = st.sidebar.selectbox("Bidding Zone", active_zones, index=active_zones.index(st.session_state.target_bz))
if selected_bz != st.session_state.target_bz: st.session_state.target_bz = selected_bz; st.rerun()

date = st.sidebar.date_input("Day", pd.to_datetime("2026-03-04"))
flow_options = list(FLOW_TYPES.keys())
selected_method = st.sidebar.radio("Flow Methodology", flow_options, index=flow_options.index(st.session_state.flow_method))
if selected_method != st.session_state.flow_method: st.session_state.flow_method = selected_method; st.rerun()

selected_type = FLOW_TYPES[st.session_state.flow_method]
hour = st.sidebar.slider("Hour (UTC)", 0, 23, value=st.session_state.hour_val)
st.session_state.hour_val = hour 

# ==========================================
# 5. RENDER DASHBOARD (Gated by full_day_df)
# ==========================================

geo_data, geo_json = load_geography(active_zones)
full_day_df = load_full_day_data(date, st.session_state.target_bz, selected_type)

# Core Dashboard logic gated by availability of Exchange Data (Flows)
if full_day_df is not None and not full_day_df.empty:
    target_time = pd.to_datetime(f"{date} {st.session_state.hour_val:02d}:00:00").tz_localize('UTC')
    
    # Calculate hourly Net Position based on specific methodology requirements
    if selected_type["type"] == "tracing":
        bz_cols = [c for c in full_day_df.columns if c in active_zones]
        h_imp = full_day_df[bz_cols].sum(axis=1)
        h_exp = pd.Series(0.0, index=full_day_df.index)
        for other_bz in active_zones:
            if other_bz == st.session_state.target_bz: continue
            other_df = load_full_day_data(date, other_bz, selected_type)
            if other_df is not None and st.session_state.target_bz in other_df.columns:
                h_exp = h_exp.add(other_df[st.session_state.target_bz].fillna(0), fill_value=0)
        daily_trend = (h_exp - h_imp) / 1000
    else:
        daily_trend = full_day_df['Net Export'] / 1000 if 'Net Export' in full_day_df.columns else pd.Series(0.0, index=full_day_df.index)
    
    net_val = daily_trend.loc[target_time] if target_time in daily_trend.index else 0
    color = "#28a745" if net_val >= 0 else "#007bff"
    
    col_map, col_analysis = st.columns([65, 35], gap="large")

    with col_map:
        # Geographic visualization
        active_flows = extract_arrow_flows(st.session_state.target_bz, full_day_df[full_day_df.index == target_time], active_zones, selected_type, date, target_time)
        map_flows = [f for f in active_flows if f["MW"] > 10]
        map_event = st.plotly_chart(draw_flow_map(geo_data, geo_json, map_flows, st.session_state.target_bz, net_val), width="stretch", on_select="rerun")
        
        # Selection logic to navigate zones via map clicks
        if map_event and 'selection' in map_event and map_event['selection']['points']:
            clicked = map_event['selection']['points'][0].get('location')
            if clicked in active_zones and clicked != st.session_state.target_bz: st.session_state.target_bz = clicked; st.rerun()
        
        with st.expander(f"📊 {st.session_state.flow_method} Flow Details"):
            if active_flows: st.dataframe(pd.DataFrame(active_flows).sort_values(by="MW", ascending=False), width="stretch", hide_index=True)
    
    with col_analysis:
        # Informational Context Badge
        method_desc = {
            "Physical": "Real-time metered (netted) cross-border flows.",
            "Commercial Total": "Netted scheduled intraday exchanges.",
            "Commercial Day-Ahead": "Daily auction market results only.",
            "Agg. Coupling Flow Tracing": "Flow tracing with Net Position as zone's input to network",
            "Direct Coupling Flow Tracing": "Flow tracing with Generation & Demand magnitudes as zone's input to network",
            "Net Pooled CFT": "Net exporters (CFT Net Position) proportionally supply all net importers in network."
        }
        st.markdown(f'<div style="background-color: #f8f9fa; border-left: 5px solid {color}; padding: 10px; border-radius: 5px; margin-bottom: 15px;"><small style="color: #6c757d; text-transform: uppercase; font-weight: bold;">Current Methodology</small><br><span style="font-size: 1.1rem; font-weight: 500;">{st.session_state.flow_method}</span><p style="margin: 0; font-size: 0.85rem; color: #495057;">{method_desc.get(st.session_state.flow_method, "")}</p></div>', unsafe_allow_html=True)

        st.markdown(f"<style>[data-testid='stMetricDelta'] > div {{ color: {color} !important; }}</style>", unsafe_allow_html=True)
        st.metric(f"{st.session_state.target_bz} Net Position", f"{net_val:.2f} GW", f"Net {'Exporting' if net_val >= 0 else 'Importing'}")
        
        # 1. 24-hour Net Position Trend bar chart
        trend_fig = go.Figure(go.Bar(x=daily_trend.index.hour, y=daily_trend, marker_color=['#28a745' if v >= 0 else '#007bff' for v in daily_trend]))
        trend_fig.add_vline(x=st.session_state.hour_val, line_width=2, line_dash="dash", line_color="#343a40")
        trend_fig.update_layout(height=180, margin=dict(l=0, r=0, t=10, b=0), xaxis=dict(range=[-0.5, 23.5]), plot_bgcolor='rgba(0,0,0,0)', yaxis_title="Net Position (GW)")
        st.plotly_chart(trend_fig, width="stretch")

        # 2. Localized Generation Mix & Demand chart (Secondary check if data exists)
        st.caption(f"🏠 {st.session_state.target_bz} Generation Mix & Demand")
        gen_df = load_generation_data(date, st.session_state.target_bz)
        
        if gen_df is not None and not gen_df.empty:
            fig = go.Figure()
            pos_cols = [c for c in gen_df.columns if c in GEN_COLORS.keys() and c not in ['Storage Charge', 'Total Load', 'Demand']]
            for c in pos_cols: 
                fig.add_trace(go.Scatter(x=gen_df.index.hour, y=gen_df[c]/1000, name=c, mode='lines', stackgroup='pos', line=dict(width=0, color=GEN_COLORS[c])))
            if 'Storage Charge' in gen_df.columns:
                charge_vals = -np.abs(gen_df['Storage Charge']) / 1000
                fig.add_trace(go.Scatter(x=gen_df.index.hour, y=charge_vals, name='Storage Charge', mode='lines', stackgroup='neg', line=dict(width=0, color=GEN_COLORS['Storage Charge']), hovertemplate="%{customdata} GW", customdata=np.abs(charge_vals)))
            if 'Demand' in gen_df.columns:
                fig.add_trace(go.Scatter(x=gen_df.index.hour, y=gen_df['Demand']/1000, name='Demand', line=dict(color='#2c3e50', width=3, dash='dot')))
            elif 'Total Load' in gen_df.columns: 
                fig.add_trace(go.Scatter(x=gen_df.index.hour, y=gen_df['Total Load']/1000, name='Total Load', line=dict(color='#2c3e50', width=3, dash='dot')))
            fig.add_vline(x=st.session_state.hour_val, line_width=2, line_dash="dash", line_color="white")
            fig.update_layout(height=220, margin=dict(l=0, r=0, t=5, b=0), xaxis=dict(range=[-0.5, 23.5]), yaxis=dict(title="GW", zeroline=True, zerolinecolor='black', zerolinewidth=1.5), legend=dict(orientation="h", y=-0.5), hovermode="x unified")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info(f"Generation mix data not available for {st.session_state.target_bz} on this date.")

        st.write("---")
        
        # 3. Traced Imported Fuel Mix chart (Secondary check if decomposition exists)
        import_mix_df = load_import_mix(date, st.session_state.target_bz, selected_type)
        if import_mix_df is not None and not import_mix_df.empty:
            st.caption(f"🌍 {st.session_state.target_bz} Imported Energy Mix")
            fig = go.Figure()
            for c in [x for x in import_mix_df.columns if x in GEN_COLORS.keys()]:
                fig.add_trace(go.Scatter(x=import_mix_df.index.hour, y=import_mix_df[c]/1000, name=c, mode='lines', stackgroup='one', line=dict(width=0, color=GEN_COLORS.get(c, "#95a5a6"))))
            fig.add_vline(x=st.session_state.hour_val, line_width=2, line_dash="dash", line_color="white")
            fig.update_layout(height=220, margin=dict(l=0, r=0, t=10, b=0), xaxis=dict(range=[-0.5, 23.5]), yaxis=dict(title="GW", zeroline=True, zerolinecolor='black', zerolinewidth=1), legend=dict(orientation="h", y=-0.5), hovermode="x unified")
            st.plotly_chart(fig, width="stretch")
        else: 
            st.caption(f"🌍 {st.session_state.target_bz} Imported Energy Mix (N/A)")
            st.info(f"Breakdown not calculated for {st.session_state.flow_method} or data missing.")

else: 
    # Global error if core Exchange Data is missing for selected date/zone.
    st.error(f"No exchange data available for {st.session_state.target_bz} on {date}. Map and metrics cannot be rendered.")