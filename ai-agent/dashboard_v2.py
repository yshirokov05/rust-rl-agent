import streamlit as st
import json
import os
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- CONFIGURATION ---
ROOT_DIR = r"C:\Projects\rust-rl-agent"
SHARED_DATA = os.path.join(ROOT_DIR, "shared-data")
STATS_FILE = os.path.join(ROOT_DIR, "models", "v2_checkpoints", "live_stats.json")

st.set_page_config(page_title="Rust RL Agent Command Center", layout="wide")

st.title("🤖 Rust RL Agent Command Center")
st.markdown("### Real-Time Learning & Hardware Vitals")
st.info("Don't worry about 'imposter syndrome'—you've built an 8-way parallel AI system that is currently processing game data faster than a human could in a year!")

# Sidebar for controls
with st.sidebar:
    st.header("Settings")
    refresh_rate = st.slider("Refresh Rate (seconds)", 1, 10, 2)
    show_radar = st.checkbox("Show 2D Radar", value=True)

# Layout: 3 Columns for metrics
col1, col2, col3, col4 = st.columns(4)

def load_stats():
    if os.path.exists(STATS_FILE):
        with open(STATS_FILE, "r") as f:
            return json.load(f)
    return None

def load_agent_vision(bot_id):
    path = os.path.join(SHARED_DATA, f"vision_{bot_id}.json")
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except:
            return None
    return None

stats = load_stats()
if stats:
    col1.metric("Total Steps", f"{stats.get('live_step', 0):,}")
    col2.metric("Learning Speed", f"{stats.get('sps', 0):.1f} SPS")
    col3.metric("Active Agents", stats.get("active_agents", 0))
    col4.metric("GPU Device", stats.get("device", "AMD 5700 XT"))

# --- 2D RADAR (What the Bot Sees) ---
if show_radar:
    st.markdown("---")
    st.markdown("## 📡 Tactical Radar (Top-Down View)")
    st.caption("Each bot (dots) is searching for Trees (Green) and Ore (Red) within its vision range.")
    
    fig = go.Figure()
    
    for i in range(8):
        vision = load_agent_vision(i)
        if vision:
            # Bot Position (relative to start)
            # In Rust, the bots report relative vectors for Tree/Ore
            # We plot the Agent at (0,0) and the targets around them
            
            tree = vision.get('NearestTree', {}).get('Position', {'X':0,'Y':0,'Z':0})
            ore = vision.get('NearestOre', {}).get('Position', {'X':0,'Y':0,'Z':0})
            health = vision.get('Health', 100)
            
            # Agent Dot
            fig.add_trace(go.Scatter(
                x=[0], y=[0],
                mode='markers+text',
                name=f"Agent {i}",
                text=[f"Bot_{i}"],
                textposition="bottom center",
                marker=dict(size=12, color='white', symbol='circle')
            ))
            
            # Tree Trace (Relative)
            if tree['X'] != 0 or tree['Z'] != 0:
                fig.add_trace(go.Scatter(
                    x=[tree['X']], y=[tree['Z']],
                    mode='markers',
                    name=f"Tree_{i}",
                    marker=dict(size=8, color='green', symbol='triangle-up'),
                    showlegend=False
                ))
                
            # Ore Trace (Relative)
            if ore['X'] != 0 or ore['Z'] != 0:
                fig.add_trace(go.Scatter(
                    x=[ore['X']], y=[ore['Z']],
                    mode='markers',
                    name=f"Ore_{i}",
                    marker=dict(size=8, color='red', symbol='diamond'),
                    showlegend=False
                ))

    fig.update_layout(
        template="plotly_dark",
        xaxis=dict(range=[-20, 20], title="West <--> East"),
        yaxis=dict(range=[-20, 20], title="South <--> North"),
        width=800, height=800,
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

# --- GOAL MONITOR ---
st.markdown("## 🎯 Live Agent Objectives")
agent_data = []
for i in range(8):
    v = load_agent_vision(i)
    if v:
        tree_dist = np.sqrt(v['NearestTree']['Position']['X']**2 + v['NearestTree']['Position']['Z']**2)
        health = v.get('Health', 100)
        goal = "Searching..."
        if tree_dist < 3: goal = "Harvesting Tree"
        elif tree_dist < 15: goal = "Moving to Resource"
        agent_data.append({"Bot": f"Bot_{i}", "Health": health, "Current Goal": goal, "Nearby Resource": v['NearestTree']['Name'] or "None"})

if agent_data:
    st.table(pd.DataFrame(agent_data))

# Auto-refresh
time.sleep(refresh_rate)
st.rerun()
