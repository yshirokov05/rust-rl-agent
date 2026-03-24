import os
import json
import time
import numpy as np
from torchvision.utils import save_image
import torch

DEPTH_LOG_DIR = r"C:\Projects\rust-rl-agent\shared-data\depth_logs"
VISION_PATH = r"C:\Projects\rust-rl-agent\shared-data\vision_0.json"
SHARED_DATA_DIR = r"C:\Projects\rust-rl-agent\shared-data"

def calculate_sps():
    files = [f for f in os.listdir(DEPTH_LOG_DIR) if f.endswith('.npz')]
    if len(files) < 1002:
        return 0.0, len(files)
    
    # Sort by timestamp in filename
    def get_ts(f):
        try:
            return int(f.split('_')[1].split('.')[0])
        except:
            return 0
    
    files.sort(key=get_ts)
    last_files = files[-1001:]
    
    ts_start = get_ts(last_files[0])
    ts_end = get_ts(last_files[-1])
    
    duration_ms = ts_end - ts_start
    if duration_ms == 0:
        return 0.0, 1000
    
    sps = 1000.0 / (duration_ms / 1000.0)
    return sps, 1000

def capture_vision():
    if not os.path.exists(VISION_PATH):
        return "ERROR: vision_0.json not found"
    
    with open(VISION_PATH, 'r') as f:
        data = json.load(f)
    
    matrix = data.get('DepthMatrix', [])
    if not matrix or len(matrix) != 7056:
        return f"ERROR: Invalid matrix size ({len(matrix)})"
    
    # Convert to 84x84 tensor
    img_tensor = torch.tensor(matrix).view(1, 84, 84)
    save_path = os.path.join(SHARED_DATA_DIR, "live_vision.png")
    save_image(img_tensor, save_path)
    return f"SUCCESS: Saved to {save_path}"

def audit_rewards():
    total_gathered = 0
    bot_counts = 8
    samples = 20 # Check last 20 frames per bot
    
    sulfur_found = False
    
    for i in range(bot_counts):
        p = os.path.join(SHARED_DATA_DIR, f"vision_{i}.json")
        if os.path.exists(p):
            with open(p, 'r') as f:
                d = json.load(f)
                if d.get('HasGathered'):
                    total_gathered += 1
                if "sulfur" in d.get('NearestOre', {}).get('Name', '').lower():
                    sulfur_found = True
                    
    return total_gathered, sulfur_found

if __name__ == "__main__":
    sps, count = calculate_sps()
    vision_status = capture_vision()
    gathered, sulfur = audit_rewards()
    
    print(f"--- LIVE STATUS REPORT ---")
    print(f"SPS: {sps:.2f} (Calculated over last {count} steps)")
    print(f"Vision: {vision_status}")
    print(f"Rewards: {gathered} gathering events detected in current snapshot.")
    print(f"Sulfur Target: {'DETECTED' if sulfur else 'NOT SEEN'}")
