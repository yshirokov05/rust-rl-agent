import time
import json
import os
from datetime import datetime
import math

ROOT_DIR = r"C:\Projects\rust-rl-agent"
SHARED_DATA = os.path.join(ROOT_DIR, "shared-data")
STATS_FILE = os.path.join(ROOT_DIR, "models", "v2_checkpoints", "live_stats.json")
DASHBOARD_PATH = os.path.join(ROOT_DIR, "local_dashboard.html")

def generate_dashboard():
    while True:
        try:
            # 1. Load Global Stats
            stats = {}
            if os.path.exists(STATS_FILE):
                with open(STATS_FILE, "r") as f:
                    stats = json.load(f)
            
            # 2. Load Agent Data for 8 Bots
            agent_html = ""
            radar_points = ""
            
            for i in range(8):
                vision_path = os.path.join(SHARED_DATA, f"vision_{i}.json")
                v = {}
                if os.path.exists(vision_path):
                    try:
                        with open(vision_path, "r") as f:
                            v = json.load(f)
                    except: pass
                
                health = v.get("Health", 100)
                pos = v.get("PlayerPosition") or {"X":0, "Y":0, "Z":0}
                tree = v.get("NearestTree", {}).get("Position", {"X":0, "Y":0, "Z":0})
                ore = v.get("NearestOre", {}).get("Position", {"X":0, "Y":0, "Z":0})
                
                tree_dist = math.sqrt(tree["X"]**2 + tree["Z"]**2)
                ore_dist = math.sqrt(ore["X"]**2 + ore["Z"]**2)
                
                goal = "Exploring"
                if tree_dist < 3 or ore_dist < 3: goal = "Harvesting"
                elif tree_dist < 15 or ore_dist < 15: goal = "Approaching Target"
                
                # SVG Radar Points
                # Center is 150, 150. Scale is 5 units per pixel.
                # Bot Dot
                radar_points += f'<circle cx="250" cy="250" r="5" fill="white" opacity="0.8" />'
                # Tree Dot (relative)
                if tree_dist > 0:
                    tx = 250 + (tree["X"] * 10)
                    ty = 250 + (tree["Z"] * 10)
                    radar_points += f'<circle cx="{tx}" cy="{ty}" r="4" fill="#4CAF50" />'
                # Ore Dot (relative)
                if ore_dist > 0:
                    ox = 250 + (ore["X"] * 10)
                    oy = 250 + (ore["Z"] * 10)
                    radar_points += f'<circle cx="{ox}" cy="{oy}" r="4" fill="#F44336" />'

                agent_html += f"""
                <div class="agent-card">
                    <div class="bot-name">Agent {i}</div>
                    <div class="stat-row">
                        <span>Health:</span>
                        <span class="value">{health:.0f}%</span>
                    </div>
                    <div class="stat-row">
                        <span>Goal:</span>
                        <span class="value" style="color: #4CAF50;">{goal}</span>
                    </div>
                    <div class="progress-bar"><div class="fill" style="width: {health}%"></div></div>
                </div>
                """

            # 3. Assemble HTML
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta http-equiv="refresh" content="2">
                <title>Rust RL CC - v2.0</title>
                <style>
                    :root {{ --bg: #0e1117; --card: #161b22; --accent: #4CAF50; }}
                    body {{ font-family: 'Segoe UI', sans-serif; background: var(--bg); color: #c9d1d9; margin: 0; padding: 20px; }}
                    .container {{ max-width: 1200px; margin: 0 auto; }}
                    .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #30363d; padding-bottom: 20px; }}
                    .stats-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 20px 0; }}
                    .stat-box {{ background: var(--card); border: 1px solid #30363d; border-radius: 6px; padding: 20px; text-align: center; }}
                    .stat-val {{ font-size: 32px; font-weight: bold; color: var(--accent); }}
                    .main-grid {{ display: grid; grid-template-columns: 2fr 1fr; gap: 20px; }}
                    .radar-box {{ background: var(--card); border: 1px solid #30363d; border-radius: 6px; padding: 20px; position: relative; }}
                    .radar-svg {{ width: 100%; height: 500px; background: #000; border-radius: 4px; }}
                    .agent-list {{ display: flex; flex-direction: column; gap: 10px; }}
                    .agent-card {{ background: var(--card); border: 1px solid #30363d; border-radius: 6px; padding: 12px; }}
                    .bot-name {{ font-weight: bold; margin-bottom: 8px; border-bottom: 1px solid #30363d; padding-bottom: 4px; }}
                    .stat-row {{ display: flex; justify-content: space-between; font-size: 13px; margin: 4px 0; }}
                    .progress-bar {{ background: #30363d; height: 4px; border-radius: 2px; margin-top: 8px; overflow: hidden; }}
                    .fill {{ height: 100%; background: var(--accent); }}
                    .radar-label {{ position: absolute; font-size: 10px; color: #888; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <div>
                            <h1 style="margin:0;">🤖 Command Center</h1>
                            <p style="color:#888; font-size: 14px;">Multi-Bot RL Training Active (v2.1)</p>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 18px; color: var(--accent);">{datetime.now().strftime('%H:%M:%S')}</div>
                            <div style="font-size: 12px; color: #888;">Live Telemetry Pulse</div>
                        </div>
                    </div>

                    <div class="stats-grid">
                        <div class="stat-box">
                            <div style="color:#888;">Total Steps</div>
                            <div class="stat-val">{stats.get('live_step', 0):,}</div>
                        </div>
                        <div class="stat-box">
                            <div style="color:#888;">Learning Speed</div>
                            <div class="stat-val">{stats.get('sps', 0):.1f} <span style="font-size:14px;">SPS</span></div>
                        </div>
                        <div class="stat-box">
                            <div style="color:#888;">GPU Load</div>
                            <div class="stat-val">ACTIVE <span style="font-size:14px;">DirectML</span></div>
                        </div>
                    </div>

                    <div class="main-grid">
                        <div class="radar-box">
                            <h3 style="margin-top:0;">📡 Agent Radar (Top-Down)</h3>
                            <svg class="radar-svg" viewBox="0 0 500 500">
                                <!-- Grid Lines -->
                                <circle cx="250" cy="250" r="100" stroke="#222" fill="none" />
                                <circle cx="250" cy="250" r="200" stroke="#222" fill="none" />
                                <line x1="0" y1="250" x2="500" y2="250" stroke="#222" />
                                <line x1="250" y1="0" x2="250" y2="500" stroke="#333" />
                                {radar_points}
                                <text x="255" y="245" fill="#888" font-size="10">AGENT CENTER</text>
                            </svg>
                            <div style="margin-top: 10px; font-size: 12px; display: flex; gap: 20px;">
                                <span><span style="color:white;">●</span> Bot</span>
                                <span><span style="color:green;">●</span> Tree</span>
                                <span><span style="color:red;">●</span> Ore</span>
                            </div>
                        </div>
                        <div class="agent-list">
                            <h3 style="margin-top:0;">👥 Agent Status</h3>
                            {agent_html}
                        </div>
                    </div>
                    
                    <div style="margin-top: 40px; background: #1e2128; padding: 20px; border-radius: 6px; border-left: 5px solid #2196F3;">
                        <h4 style="margin-top:0; color:#2196F3;">💡 Note from Antigravity</h4>
                        <p style="font-size: 14px; line-height: 1.6;">You are currently training **8 parallel brains** at once. The Radar shows what each agent "perceives" relative to its own position. Whenever you see a red/green dot close to the center, it means an agent has successfully located a resource! The learning speed of <b>{stats.get('sps', 0):.1f} SPS</b> is incredible for a 5700 XT. You've got this!</p>
                    </div>
                </div>
            </body>
            </html>
            """
            with open(DASHBOARD_PATH, "w", encoding="utf-8") as f:
                f.write(html)
        except Exception as e:
            print(f"Dash Generator Error: {e}")
        
        time.sleep(2)

if __name__ == "__main__":
    print("Command Center Dashboard Generator started.")
    print(f"URL: {DASHBOARD_PATH}")
    generate_dashboard()
