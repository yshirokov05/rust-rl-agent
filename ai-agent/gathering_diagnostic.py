import os
import json
import time
import re
import numpy as np

LOG_PATH = r"C:\rust_research\carbon\logs\Carbon.Core.log"
VISION_BASE = r"C:\Projects\rust-rl-agent\shared-data\vision_{0}.json"
DEPTH_LOG_DIR = r"C:\Projects\rust-rl-agent\shared-data\depth_logs"
DURATION_SEC = 15 * 60
INTERVAL = 10 

def get_dist(json_path):
    if not os.path.exists(json_path): return 100.0
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        ore = data.get('NearestOre', {}).get('Position') or {'X': 100.0, 'Y': 100.0, 'Z': 100.0}
        return (ore['X']**2 + ore['Y']**2 + ore['Z']**2)**0.5
    except:
        return 100.0

def count_successes():
    if not os.path.exists(LOG_PATH): return 0
    try:
        with open(LOG_PATH, 'r') as f:
            content = f.read()
        return len(re.findall(r"GATHER_SUCCESS", content))
    except:
        return 0

def get_sps():
    files = [f for f in os.listdir(DEPTH_LOG_DIR) if f.endswith('.npz')]
    if len(files) < 101: return 0.0
    
    files.sort(key=lambda x: os.path.getmtime(os.path.join(DEPTH_LOG_DIR, x)))
    last_files = files[-101:]
    
    def extract_ts(f):
        try: return int(f.split('_')[1].split('.')[0])
        except: return 0
        
    ts_start = extract_ts(last_files[0])
    ts_end = extract_ts(last_files[-1])
    
    duration = (ts_end - ts_start) / 1000.0
    if duration <= 0: return 0.0
    return 100.0 / duration

if __name__ == "__main__":
    print(f"--- GATHERING AUDIT STARTED (Target: 15m) ---")
    start_time = time.time()
    initial_successes = count_successes()
    triggers = 0
    
    try:
        while time.time() - start_time < DURATION_SEC:
            current_successes = count_successes() - initial_successes
            
            for i in range(8):
                if get_dist(VISION_BASE.format(i)) < 2.0:
                    triggers += 1
            
            sps = get_sps()
            elapsed = int(time.time() - start_time)
            success_rate = (current_successes / (triggers/8) * 100) if triggers > 0 else 0
            
            print(f"[{elapsed//60:02d}:{elapsed%60:02d}] SPS: {sps:.2f} | Triggers: {triggers} | Successes: {current_successes} | Rate: {success_rate:.1f}%")
            time.sleep(INTERVAL)
            
    except KeyboardInterrupt:
        pass

    final_successes = count_successes() - initial_successes
    final_rate = (final_successes / (triggers/8) * 100) if triggers > 0 else 0
    print(f"--- AUDIT COMPLETE ---")
    print(f"Total Triggers: {triggers}")
    print(f"Total Successes: {final_successes}")
    print(f"Final Success Rate: {final_rate:.1f}%")
    print(f"Final Status: {'VETERAN - TRANSITIONING TO 8-HOUR RUN' if final_rate > 50 else 'ADJUSTMENT REQUIRED'}")
