import json
import os
import re
import subprocess
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

PROJECT_ROOT = "c:\\Projects\\rust-rl-agent"
VISION_PATH = os.path.join(PROJECT_ROOT, "shared-data", "vision.json")
LOG_PATH = os.path.join(PROJECT_ROOT, "train_v1_2.log")
STATE_PATH = os.path.join(PROJECT_ROOT, "project_state.json")

def parse_latest_stats():
    if not os.path.exists(LOG_PATH):
        return {}
    
    try:
        with open(LOG_PATH, "r") as f:
            lines = f.readlines()
        
        stats = {}
        table_found = False
        for line in reversed(lines):
            if "|" in line:
                table_found = True
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if len(parts) >= 2:
                    key, value = parts[0], parts[1]
                    if key not in stats:
                        stats[key] = value
            elif table_found:
                if "iterations" in stats and "ep_rew_mean" in stats:
                    break
        return stats
    except:
        return {}

def get_last_logs(n=20):
    if not os.path.exists(LOG_PATH):
        return "Waiting for agent to start..."
    try:
        with open(LOG_PATH, "r") as f:
            lines = f.readlines()
            # Only show lines containing "Groundbreaker" or errors
            filtered = [l for l in lines if "Groundbreaker" in l or "Error" in l or "Traceback" in l]
            return "".join(filtered[-n:])
    except:
        return ""

@app.route("/command/<cmd>", methods=["POST"])
def run_command(cmd):
    try:
        if cmd == "respawn":
            # Just delete the actions file to force a reset if the bot is stuck or Kill the bot?
            # Actually, the plugin respawns if _bot is dead.
            # We can't easily kill the bot from here without RCON or a flag.
            # Let's use a flag file.
            with open(os.path.join(PROJECT_ROOT, "shared-data", "command.json"), "w") as f:
                json.dump({"action": "respawn"}, f)
            return jsonify({"status": "Respawn requested"})
        
        if cmd == "sync":
            os.chdir(PROJECT_ROOT)
            subprocess.run(["git", "add", "."], shell=True)
            subprocess.run(["git", "commit", "-m", "Remote manual sync"], shell=True)
            subprocess.run(["git", "push", "origin", "main"], shell=True)
            return jsonify({"status": "Sync initiated"})
            
        return jsonify({"error": "Unknown command"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/data")
def get_data():
    data = {
        "vision": {},
        "stats": parse_latest_stats(),
        "state": {},
        "logs": get_last_logs(15)
    }
    
    if os.path.exists(VISION_PATH):
        try:
            with open(VISION_PATH, "r") as f:
                data["vision"] = json.load(f)
        except:
            pass

    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, "r") as f:
                data["state"] = json.load(f)
        except:
            pass
            
    return jsonify(data)

@app.route("/")
def index():
    try:
        path = os.path.join(PROJECT_ROOT, "ai-agent", "dashboard.html")
        return open(path, encoding='utf-8').read()
    except Exception as e:
        return f"Dashboard UI Error: {str(e)}", 500

if __name__ == "__main__":
    app.run(port=8080, debug=False)
