import time
import json
import os
from datetime import datetime

ROOT_DIR = "c:\\Projects\\rust-rl-agent"
VISION_PATH = os.path.join(ROOT_DIR, "shared-data", "vision.json")
ACTIONS_PATH = os.path.join(ROOT_DIR, "shared-data", "actions.json")
DASHBOARD_PATH = os.path.join(ROOT_DIR, "local_dashboard.html")

def generate_dashboard():
    while True:
        try:
            vision = {}
            if os.path.exists(VISION_PATH):
                try:
                    with open(VISION_PATH, 'r') as f:
                        vision = json.load(f)
                except:
                    pass
            
            actions = {}
            if os.path.exists(ACTIONS_PATH):
                try:
                    with open(ACTIONS_PATH, 'r') as f:
                        actions = json.load(f)
                except:
                    pass

            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta http-equiv="refresh" content="1">
                <title>Rust RL Local Dashboard</title>
                <style>
                    body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0e1117; color: white; padding: 20px; }}
                    .card {{ background: #1e2128; border-radius: 10px; padding: 20px; margin-bottom: 20px; border-left: 5px solid #4CAF50; }}
                    .metric {{ font-size: 24px; font-weight: bold; color: #4CAF50; }}
                    .label {{ font-size: 14px; color: #888; }}
                    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }}
                    .raw {{ background: #000; padding: 10px; border-radius: 5px; font-family: monospace; font-size: 12px; overflow: auto; max-height: 200px; }}
                </style>
            </head>
            <body>
                <h1>Rust RL Real-Time Monitor</h1>
                <p>Last Update: {datetime.now().strftime('%H:%M:%S')}</p>
                
                <div class="grid">
                    <div class="card">
                        <div class="label">Health</div>
                        <div class="metric">{vision.get('Health', 'N/A')}%</div>
                    </div>
                    <div class="card" style="border-left-color: #2196F3;">
                        <div class="label">Position</div>
                        <div class="metric">X: {(vision.get('PlayerPosition') or {}).get('X',0):.1f} Z: {(vision.get('PlayerPosition') or {}).get('Z',0):.1f}</div>
                    </div>
                    <div class="card" style="border-left-color: #FFEB3B;">
                        <div class="label">Harvesting</div>
                        <div class="metric">{'YES' if actions.get('Attack') else 'NO'}</div>
                    </div>
                </div>

                <h3>Vision (Raw)</h3>
                <div class="raw">{json.dumps(vision, indent=2)}</div>
                
                <h3>Actions (Raw)</h3>
                <div class="raw">{json.dumps(actions, indent=2)}</div>
            </body>
            </html>
            """
            with open(DASHBOARD_PATH, 'w', encoding='utf-8') as f:
                f.write(html)
        except Exception as e:
            print(f"Dash Error: {e}")
        
        time.sleep(1)

if __name__ == "__main__":
    print("Dashboard generator started. Open 'local_dashboard.html' in your browser.")
    generate_dashboard()
