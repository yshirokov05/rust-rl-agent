import os
import time
import subprocess
import json

# MASTER GOVERNOR: Phase F (3D Vision + 60HP Constraint)
SHARED_DATA = "C:/Projects/rust-rl-agent/shared-data/vision_0.json"
REBOOT_THRESHOLD = 300 # 5 minutes of stale data = reboot

def check_heartbeat():
    if not os.path.exists(SHARED_DATA):
        return False
    mtime = os.path.getmtime(SHARED_DATA)
    return (time.time() - mtime) < REBOOT_THRESHOLD

def enforce_hp_governor():
    if os.path.exists(SHARED_DATA):
        try:
            with open(SHARED_DATA, 'r') as f:
                data = json.load(f)
                hp = data.get("Health", 100)
                if hp < 60:
                    print(f"CRITICAL: HP dropped to {hp}. Triggering Emergency Respawn.")
                    # In a real scenario, we'd send a command to the server
                    # For now, we log it for the agent to see
        except Exception as e:
            print(f"Governor Err: {e}")

if __name__ == "__main__":
    print("Master Governor Active (CNN Mode). Monitoring Heartbeat...")
    while True:
        if not check_heartbeat():
            print("ALERT: Server Heartbeat Lost. Initiating Fail-Safe Reboot...")
            # subprocess.run(["powershell", "Restart-Service RustDedicated"]) # Example
        
        enforce_hp_governor()
        time.sleep(10)
