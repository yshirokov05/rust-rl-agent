import os
import time
import glob

DATA_DIR = r"C:\Projects\rust-rl-agent\shared-data\depth_logs"

def monitor():
    print(f"Monitoring {DATA_DIR} for [HARD COUNT]...")
    while True:
        try:
            # Count all .npz files recursively
            files = glob.glob(os.path.join(DATA_DIR, "**", "batch_*.npz"), recursive=True)
            batch_count = len(files)
            step_count = batch_count * 200
            
            print(f"[{time.strftime('%H:%M:%S')}] [HARD COUNT]: {step_count} synced steps ({batch_count} batches).")
        except Exception as e:
            print(f"Monitor Error: {e}")
            
        time.sleep(60)

if __name__ == "__main__":
    monitor()
