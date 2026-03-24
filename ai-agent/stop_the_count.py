import os
import time
import glob
import subprocess

DATA_DIR = r"C:\Projects\rust-rl-agent\shared-data\depth_logs"
TARGET_BATCHES = 7500 # 1,500,000 / 200

def terminator():
    print(f"TERMINATOR ACTIVE: Watching for {TARGET_BATCHES} batches...")
    while True:
        try:
            files = glob.glob(os.path.join(DATA_DIR, "**", "batch_*.npz"), recursive=True)
            count = len(files)
            
            if count >= TARGET_BATCHES:
                print(f"[{time.strftime('%H:%M:%S')}] TARGET REACHED: {count*200} STEPS. EXECUTING STOP ORDER.")
                
                # Kill Python shards
                subprocess.run("taskkill /F /IM python.exe /T", shell=True)
                # Kill Rust Server
                subprocess.run("taskkill /F /IM RustDedicated.exe /T", shell=True)
                
                print("Swarm and Server terminated. Mission Accomplished.")
                break
                
            if count % 10 == 0:
                print(f"Current Progress: {count*200} / 1,500,000 steps...")
                
        except Exception as e:
            print(f"Terminator Error: {e}")
            
        time.sleep(2)

if __name__ == "__main__":
    terminator()
