import os
import time
import numpy as np
import psutil
import subprocess

LOG_DIR = r"C:\Projects\rust-rl-agent\shared-data\depth_logs"
TARGET_STEPS = 20000

def get_step_count():
    try:
        # Each batch file contains 50 steps
        files = [f for f in os.listdir(LOG_DIR) if f.startswith('batch_')]
        return len(files) * 50
    except:
        return 0

def kill_process(name):
    for proc in psutil.process_iter(['name']):
        if proc.info['name'] == name:
            proc.kill()

def main():
    print(f"Monitoring {LOG_DIR} for {TARGET_STEPS} DISTILLATION steps...")
    
    while True:
        count = get_step_count()
        if count >= TARGET_STEPS:
            print(f"THRESHOLD REACHED: {count} synced steps. Executing Protocol...")
            break
        print(f"Current Synced: {count} / {TARGET_STEPS}...", end='\r')
        time.sleep(10)
        
    print("Pausing Inference...")
    subprocess.run(["cmd", "/c", "taskkill", "/F", "/IM", "python.exe", "/T"])
    
    print("Stopping Rust Server to free VRAM...")
    kill_process("RustDedicated.exe")
    
    print("TRANSITIONING TO BAKE: Launching bc_nature_cnn.py...")
    subprocess.Popen(["cmd", "/c", "start", "c:\\Projects\\rust-rl-agent\\venv\\Scripts\\python.exe", "c:\\Projects\\rust-rl-agent\\ai-agent\\bc_nature_cnn.py"])
    
    print("11:48 AM PROTOCOL COMPLETE. BC Training is now ACTIVE.")

if __name__ == "__main__":
    main()
