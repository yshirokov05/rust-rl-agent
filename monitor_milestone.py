import os
import time
import json
import shutil

STATS_PATH = r"C:\Projects\rust-rl-agent\models\v2_checkpoints\live_stats.json"
CHECKPOINT_DIR = r"C:\Projects\rust-rl-agent\models\v2_checkpoints"
GOLD_DIR = r"C:\Projects\rust-rl-agent\models\GOLD_MODELS"
TARGET_STEP = 2200000
TARGET_FILE = "overnight_win_2200000_steps.zip"

def monitor():
    os.makedirs(GOLD_DIR, exist_ok=True)
    print(f"Monitoring for Milestone: {TARGET_STEP} steps...")
    
    while True:
        # Check live JSON
        if os.path.exists(STATS_PATH):
            try:
                with open(STATS_PATH, 'r') as f:
                    data = json.load(f)
                    current_step = data.get('live_step', 0)
                    print(f"[{time.strftime('%H:%M:%S')}] Current Step: {current_step}")
                    if current_step >= TARGET_STEP:
                        print("Milestone reached in live stats!")
            except Exception as e:
                print(f"Error reading stats: {e}")

        # Check for the actual file
        checkpoint_path = os.path.join(CHECKPOINT_DIR, TARGET_FILE)
        if os.path.exists(checkpoint_path):
            print(f"FOUND GOLD MODEL: {checkpoint_path}")
            dest_path = os.path.join(GOLD_DIR, TARGET_FILE)
            shutil.copy2(checkpoint_path, dest_path)
            print(f"EXPORTED TO: {dest_path}")
            
            # Create a success flag for the agent
            with open("milestone_2.2m_reached.txt", "w") as f:
                f.write(f"Reached at {time.ctime()}")
            break
            
        time.sleep(60)

if __name__ == "__main__":
    monitor()
