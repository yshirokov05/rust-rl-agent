import streamlit as st
import json
import time
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Config
st.set_page_config(page_title="OBSIDIAN RECON CENTER", layout="wide")
SHARED_DIR = r"C:\Projects\rust-rl-agent\shared-data"
STATS_PATH = r"C:\Projects\rust-rl-agent\models\v2_checkpoints\live_stats.json"

st.markdown("""
<style>
    .stApp { background: #060606; }
    .stMetric { background: #111; padding: 15px; border-radius: 10px; border: 1px solid #333; }
    h1, h2, h3 { color: #1DB954 !important; font-family: 'Inter', sans-serif; }
    .bot-active { color: #1DB954; font-weight: bold; }
    .resource-box { color: #4CAF50; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("🕶️ OBSIDIAN RECON CENTER v1.1")

def load_json(path):
    if not os.path.exists(path): return None
    try:
        with open(path, 'r') as f: return json.load(f)
    except: return None

# SETUP FIXED CONTAINERS
header = st.empty()
col_main, col_stats = st.columns([2, 1])
radar_container = col_main.empty()
stream_container = col_stats.empty()

# Persistent loop
while True:
    stats = load_json(STATS_PATH)
    
    # 1. Update Header
    if stats:
        is_stale = (time.time() - stats.get('timestamp', 0)) > 45
        status = "⚠️ SYNCING..." if is_stale else "⚡ LIVE FEED"
        header.markdown(f"### 📊 Status: {status} | Steps: {stats.get('live_step', 0):,} | SPS: {stats.get('sps', 0):.1f} | GPU: {stats.get('device', 'AMD')}")

    # 2. Update Radar and Stream
    with radar_container.container():
        st.subheader("📡 Tactical Radar (First-Person Swarm)")
        fig = go.Figure()
        
        # Grid
        for r in [5, 10, 15, 20]:
            fig.add_shape(type="circle", x0=-r, y0=-r, x1=r, y1=r, line=dict(color="#222", width=1, dash="dot"))
        
        # Plot Bot #0 (The Intelligence at center)
        fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers+text', name="Intelligence", 
                                 text=["BOT_0"], textposition="bottom center",
                                 marker=dict(size=12, color='#1DB954', symbol='circle')))
        
        # Plot Local Resources from shared-data
        for i in range(8):
            v = load_json(os.path.join(SHARED_DIR, f"vision_{i}.json"))
            if v:
                tree = v.get('NearestTree', {}).get('Position', {'X':0,'Y':0,'Z':0})
                if abs(tree['X']) > 0.01:
                    fig.add_trace(go.Scatter(x=[tree['X']], y=[tree['Z']], mode='markers', name=f"Resource {i}",
                                             marker=dict(size=14, color='#4CAF50', symbol='square')))
        
        fig.update_layout(template="plotly_dark", height=600, showlegend=False, 
                          xaxis=dict(range=[-25, 25], gridcolor="#111"), 
                          yaxis=dict(range=[-25, 25], gridcolor="#111"),
                          margin=dict(l=0,r=0,t=0,b=0))
        
        st.plotly_chart(fig, use_container_width=True, key=f"radar_{time.time()}")
        st.caption("🟢 Green Circle = Agent Intelligence | 🟩 Green Square = Harvestable Resource")

    with stream_container.container():
        st.subheader("👥 Intelligence Swarm")
        rows = []
        for i in range(8):
            v = load_json(os.path.join(SHARED_DIR, f"vision_{i}.json"))
            if v:
                t_name = v.get('NearestTree', {}).get('Name', 'None')
                t_pos = v.get('NearestTree', {}).get('Position', {'X':0,'Z':0})
                dist = np.sqrt(t_pos['X']**2 + t_pos['Z']**2)
                goal = "HARVESTING" if dist < 3 and t_name != "None" else "EXPLORING"
                rows.append({"Agent": f"#{i}", "Goal": goal, "Target": t_name[:15] if t_name != "None" else "Scanning...", "Range": f"{dist:.1f}m"})
        
        if rows: st.table(pd.DataFrame(rows))
        else: st.info("Handshaking with Python server...")

    time.sleep(2)
