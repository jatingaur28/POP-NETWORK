"""
Enterprise Network Operations Center (NOC) Command Dashboard.
Renders live hardware telemetry, Topographical AI anomalies, and Segment Routing states
using high-performance Plotly WebGL and Real-World GIS coordinates.
"""

import os
import sys
import json
import hashlib
from datetime import datetime

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import redis

# Ensure root path resolution for sdn_kdn module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from sdn_kdn.core.config import settings

# Thread-safe Synchronous Redis Connection for Dash Callbacks
r = redis.Redis(
    host=settings.redis_host, 
    port=settings.redis_port, 
    decode_responses=True,
    socket_timeout=2.0
)

# ==============================================================================
# ENTERPRISE THEME & STYLING
# ==============================================================================
NOC_THEME = {
    "bg_base": "#0A0E17",
    "bg_panel": "#111827",
    "border": "#1F2937",
    "text_main": "#F8FAFC",
    "text_muted": "#94A3B8",
    "brand_cyan": "#00E5FF",
    "status_healthy": "#10B981",
    "status_warning": "#F59E0B",
    "status_critical": "#EF4444",
}

app = dash.Dash(__name__, title="KDN Enterprise Command Center")

app.layout = html.Div(
    style={
        'backgroundColor': NOC_THEME["bg_base"], 
        'color': NOC_THEME["text_main"], 
        'fontFamily': 'Consolas, monospace', 
        'minHeight': '100vh',
        'margin': '0',
        'padding': '15px'
    },
    children=[
        # --- HEADER ---
        html.Div(
            style={
                'display': 'flex', 
                'justifyContent': 'space-between', 
                'alignItems': 'center',
                'borderBottom': f'1px solid {NOC_THEME["border"]}',
                'paddingBottom': '10px',
                'marginBottom': '15px'
            },
            children=[
                html.H2(
                    "🌐 KDN GLOBAL ORCHESTRATOR", 
                    style={'margin': '0', 'color': NOC_THEME["brand_cyan"], 'fontWeight': 'bold', 'letterSpacing': '2px'}
                ),
                html.Div(id='live-clock', style={'fontSize': '16px', 'color': NOC_THEME["text_muted"]})
            ]
        ),
        
        # --- MAIN WORKSPACE ---
        html.Div(
            style={'display': 'flex', 'gap': '15px', 'height': '85vh'},
            children=[
                
                # LEFT: Global GIS Topology Map
                html.Div(
                    style={
                        'flex': '3', 
                        'backgroundColor': NOC_THEME["bg_panel"], 
                        'border': f'1px solid {NOC_THEME["border"]}',
                        'borderRadius': '8px',
                        'position': 'relative'
                    },
                    children=[
                        dcc.Graph(
                            id='global-network-map', 
                            style={'width': '100%', 'height': '100%'},
                            config={'displayModeBar': False}
                        )
                    ]
                ),
                
                # RIGHT: AIOps & Hardware Telemetry HUD
                html.Div(
                    style={
                        'flex': '1', 
                        'display': 'flex', 
                        'flexDirection': 'column', 
                        'gap': '15px'
                    },
                    children=[
                        # Top-Right Panel: Active Anomalies & BGP Mitigations
                        html.Div(
                            style={
                                'flex': '1',
                                'backgroundColor': NOC_THEME["bg_panel"],
                                'border': f'1px solid {NOC_THEME["border"]}',
                                'borderRadius': '8px',
                                'padding': '15px',
                                'overflowY': 'auto'
                            },
                            children=[
                                html.H4("🚨 AIOps Security & VAE Events", style={'color': NOC_THEME["status_critical"], 'marginTop': '0'}),
                                html.Div(id='aiops-alert-panel', style={'fontSize': '13px'})
                            ]
                        ),
                        
                        # Bottom-Right Panel: Top Congested Hardware Links
                        html.Div(
                            style={
                                'flex': '1.5',
                                'backgroundColor': NOC_THEME["bg_panel"],
                                'border': f'1px solid {NOC_THEME["border"]}',
                                'borderRadius': '8px',
                                'padding': '15px',
                                'overflowY': 'auto'
                            },
                            children=[
                                html.H4("📡 Hardware Saturation (Top 10)", style={'color': NOC_THEME["status_warning"], 'marginTop': '0'}),
                                html.Div(id='telemetry-hud-panel', style={'fontSize': '13px'})
                            ]
                        )
                    ]
                )
            ]
        ),
        
        # --- BACKGROUND WORKERS ---
        dcc.Interval(id='telemetry-interval', interval=2000, n_intervals=0),
        html.Div(id='last-state-hash', style={'display': 'none'}, children="")
    ]
)

# ==============================================================================
# RENDER LOGIC & STATE MANAGEMENT
# ==============================================================================
@app.callback(
    [
        Output('global-network-map', 'figure'), 
        Output('aiops-alert-panel', 'children'),
        Output('telemetry-hud-panel', 'children'),
        Output('last-state-hash', 'children'),
        Output('live-clock', 'children')
    ],
    [Input('telemetry-interval', 'n_intervals')],
    [State('last-state-hash', 'children')]
)
def update_dashboard(n, previous_hash):
    clock_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    try:
        raw_topology = r.get("kdn_latest_topology")
        raw_telemetry = r.get("latest_telemetry_cache")
    except redis.ConnectionError:
        return dash.no_update, "IPC Broker Offline", "Cannot reach Redis", previous_hash, clock_str
    
    if not raw_topology or not raw_telemetry:
        return dash.no_update, "Waiting for SQL Topology...", "Waiting for Hardware Stream...", previous_hash, clock_str

    # CPU Guard: SHA-256 state hashing prevents wasteful 60fps React re-renders
    state_payload = raw_topology + raw_telemetry
    current_hash = hashlib.sha256(state_payload.encode('utf-8')).hexdigest()
    
    if current_hash == previous_hash:
        return dash.no_update, dash.no_update, dash.no_update, previous_hash, clock_str

    # Parse Live Datasets
    topology = json.loads(raw_topology)
    telemetry_list = json.loads(raw_telemetry)
    telemetry_map = {t['link']: t for t in telemetry_list}
    
    fig = go.Figure()
    alert_elements = []
    hud_elements = []

    # Prepare sorting list for the Saturation HUD
    link_saturation_data = []

    # ==========================================================================
    # 1. DRAW PHYSICAL FIBER LINKS (EDGES)
    # ==========================================================================
    for lid, ldata in topology.get("links", {}).items():
        src_id = ldata.get("source") or ldata.get("src")
        dst_id = ldata.get("target") or ldata.get("dst")
        
        src_node = topology["nodes"].get(src_id)
        dst_node = topology["nodes"].get(dst_id)
        
        if not src_node or not dst_node: 
            continue
            
        # Hardware Metrics Extraction
        live_t = telemetry_map.get(lid, {})
        max_cap = float(ldata.get("max_bw_gbps", 10.0))
        live_bw = float(live_t.get("bandwidth_gbps", 0.0))
        residual_bw = max(0.0, max_cap - live_bw)
        util_pct = (live_bw / max(0.001, max_cap)) * 100.0
        
        is_anomaly = live_t.get("is_anomaly", False)

        # Dynamic Color Coding
        if is_anomaly or util_pct >= 90.0:
            color = NOC_THEME["status_critical"]
            width = 3.5
            alert_elements.append(html.Div(f"CRITICAL: {lid} operating at {util_pct:.1f}% capacity.", style={'marginBottom': '5px'}))
        elif util_pct >= 70.0:
            color = NOC_THEME["status_warning"]
            width = 2.5
        else:
            color = NOC_THEME["brand_cyan"]
            width = 1.0

        # Register for side panel sorting
        link_saturation_data.append({"id": lid, "util": util_pct, "live_bw": live_bw, "color": color})

        # Tooltip formatting
        hover_text = (
            f"<b>Link: {lid}</b><br>"
            f"Hardware Status: {'ANOMALOUS' if is_anomaly else 'HEALTHY'}<br>"
            f"Throughput: {live_bw:.2f} Gbps / {max_cap:.0f} Gbps ({util_pct:.1f}%)<br>"
            f"Residual Capacity (CSPF): {residual_bw:.2f} Gbps<br>"
            f"Active OSPF Cost: {live_t.get('ospf_cost', ldata.get('ospf_cost', 10))}<br>"
            f"Optical Temp: {live_t.get('optical_temp_c', 'N/A')} °C"
        )

        fig.add_trace(go.Scattergeo(
            lon=[src_node.get('lon', src_node.get('longitude', 0)), dst_node.get('lon', dst_node.get('longitude', 0))],
            lat=[src_node.get('lat', src_node.get('latitude', 0)), dst_node.get('lat', dst_node.get('latitude', 0))],
            mode='lines',
            line=dict(width=width, color=color),
            hoverinfo='text',
            text=hover_text,
            showlegend=False,
            opacity=0.8
        ))

    # ==========================================================================
    # 2. DRAW ROUTER NODES
    # ==========================================================================
    lons, lats, node_texts, node_colors, node_sizes = [], [], [], [], []
    
    for nid, ndata in topology.get("nodes", {}).items():
        lons.append(ndata.get('lon', ndata.get('longitude', 0)))
        lats.append(ndata.get('lat', ndata.get('latitude', 0)))
        
        is_tier_1 = int(ndata.get('tier', 2)) == 1
        node_sizes.append(14 if is_tier_1 else 8)
        node_colors.append(NOC_THEME["status_warning"] if is_tier_1 else NOC_THEME["status_healthy"])
        
        node_texts.append(
            f"<b>Router: {nid}</b><br>"
            f"Loopback IP: {ndata.get('loopback_ip', 'N/A')}<br>"
            f"Tier: {ndata.get('tier', 2)}"
        )

    fig.add_trace(go.Scattergeo(
        lon=lons, 
        lat=lats,
        mode='markers',
        marker=dict(size=node_sizes, color=node_colors, line=dict(width=1.5, color=NOC_THEME["bg_base"])),
        hovertext=node_texts,
        hoverinfo='text',
        showlegend=False
    ))

    # ==========================================================================
    # 3. GIS MAPBOX / GLOBE PROJECTION CONFIGURATION
    # ==========================================================================
    fig.update_layout(
        geo=dict(
            projection_type="orthographic", # 3D Globe Projection
            showland=True,
            landcolor="#0f172a",
            showocean=True,
            oceancolor=NOC_THEME["bg_base"],
            showcountries=True,
            countrycolor=NOC_THEME["border"],
            bgcolor=NOC_THEME["bg_base"],
            resolution=50
        ),
        paper_bgcolor=NOC_THEME["bg_panel"],
        plot_bgcolor=NOC_THEME["bg_panel"],
        margin=dict(l=0, r=0, t=0, b=0),
        dragmode="zoom"
    )

    # ==========================================================================
    # 4. POPULATE SIDE PANELS
    # ==========================================================================
    if not alert_elements:
        alert_elements = [html.Div("No active hardware anomalies. VAE models reporting optimal states.", style={'color': NOC_THEME["status_healthy"]})]

    # Sort links by utilization descending for the HUD
    link_saturation_data.sort(key=lambda x: x["util"], reverse=True)
    for link_stat in link_saturation_data[:10]:
        hud_elements.append(
            html.Div(
                style={'display': 'flex', 'justifyContent': 'space-between', 'borderBottom': f'1px solid {NOC_THEME["border"]}', 'padding': '8px 0'},
                children=[
                    html.Span(link_stat["id"], style={'fontWeight': 'bold'}),
                    html.Span(f"{link_stat['live_bw']:.1f}G ({link_stat['util']:.1f}%)", style={'color': link_stat["color"]})
                ]
            )
        )

    return fig, alert_elements, hud_elements, current_hash, clock_str

# ==============================================================================
# STANDALONE EXECUTION
# ==============================================================================
if __name__ == '__main__':
    # Binds directly to the Host/Port defined in your enterprise configuration
    logger.info("🌐 Booting Dash Enterprise NOC on %s:%s...", settings.api_host, settings.ui_port)
    app.run_server(host=settings.api_host, port=settings.ui_port, debug=False)