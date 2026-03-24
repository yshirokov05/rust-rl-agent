import zipfile
import io
import torch
import json
import os
import numpy as np
import time

CHECKPOINT_DIR = r"C:\Projects\rust-rl-agent\models\v2_checkpoints"
VISION_PATH = r"C:\Projects\rust-rl-agent\shared-data\vision_0.json"

def audit_step():
    files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".zip")]
    if not files:
        print("NO_CHECKPOINTS_FOUND")
        return
    
    # Get the latest modified file
    latest_file = max([os.path.join(CHECKPOINT_DIR, f) for f in files], key=os.path.getmtime)
    print(f"AUDITING_FILE: {latest_file}")
    
    try:
        with zipfile.ZipFile(latest_file, 'r') as archive:
            # SB3 saves metadata in 'data' file inside the zip
            with archive.open('data') as f:
                # We can't easily parse the raw SB3 data file without full SB3 load,
                # but we can try to load it with torch if it's the right format,
                # or just look at the filename which usually contains the step.
                pass
        
        # Alternative: Load with SB3 for 100% accuracy
        from stable_baselines3 import PPO
        model = PPO.load(latest_file, device="cpu")
        print(f"PHYSICAL_STEP_COUNT: {model.num_timesteps}")
    except Exception as e:
        print(f"STEP_AUDIT_ERROR: {e}")

def audit_behavior():
    if not os.path.exists(VISION_PATH):
        print(f"VISION_FILE_NOT_FOUND: {VISION_PATH}")
        return

    try:
        # Read the file multiple times to simulate "last 200 lines" (or just the current state if not log-appended)
        # Note: If vision_0.json is overwritten every tick, we can only see the current distance.
        # If it's a log, we read the end.
        with open(VISION_PATH, 'r') as f:
            data = json.load(f)
        
        player = data.get('PlayerPosition', {})
        tree = data.get('NearestTree', {}).get('Position', {})
        
        if player and tree:
            p_pos = np.array([player.get('X', 0), player.get('Y', 0), player.get('Z', 0)])
            t_pos = np.array([tree.get('X', 0), tree.get('Y', 0), tree.get('Z', 0)])
            dist = np.linalg.norm(p_pos - t_pos)
            print(f"CURRENT_DISTANCE_TO_TARGET: {dist:.2f} meters")
            
            # Check for staleness
            mtime = os.path.getmtime(VISION_PATH)
            age = time.time() - mtime
            print(f"VISION_DATA_AGE: {age:.2f} seconds")
            
            if age > 10:
                print("STATUS: SERVER_STALLED_OR_OFFLINE")
            else:
                print("STATUS: SERVER_ACTIVE")
        else:
            print("STATUS: VISION_DATA_INCOMPLETE")
            
    except Exception as e:
        print(f"BEHAVIOR_AUDIT_ERROR: {e}")

if __name__ == "__main__":
    audit_step()
    print("-" * 20)
    audit_behavior()
