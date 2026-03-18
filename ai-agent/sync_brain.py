import os
import time
import subprocess
import glob

PROJECT_ROOT = "c:\\Projects\\rust-rl-agent"
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "models", "checkpoints")
SYNC_INTERVAL = 600  # 10 minutes

def get_latest_checkpoint():
    list_of_files = glob.glob(os.path.join(CHECKPOINTS_DIR, "*.zip"))
    if not list_of_files:
        return None
    return max(list_of_files, key=os.path.getmtime)

def sync(file_path):
    try:
        print(f"[{time.ctime()}] Syncing {os.path.basename(file_path)} to GitHub...")
        os.chdir(PROJECT_ROOT)
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", "Auto_brain_sync"], check=True)
        subprocess.run(["git", "push", "origin", "main"], check=True)
        print(f"[{time.ctime()}] Sync successful!")
    except Exception as e:
        print(f"[{time.ctime()}] Sync failed: {e}")

if __name__ == "__main__":
    print(f"Improved Auto-Sync started. Watching: {CHECKPOINTS_DIR}")
    last_synced_file = None
    last_synced_mtime = 0

    while True:
        latest = get_latest_checkpoint()
        if latest:
            current_mtime = os.path.getmtime(latest)
            if latest != last_synced_file or current_mtime > last_synced_mtime:
                sync(latest)
                last_synced_file = latest
                last_synced_mtime = current_mtime
        
        time.sleep(60)
