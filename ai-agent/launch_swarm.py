import subprocess
import os
import time

PYTHON_EXE = r"C:\Projects\rust-rl-agent\venv\Scripts\python.exe"
SCRIPT_PATH = r"C:\Projects\rust-rl-agent\ai-agent\inference_v3.py"
LOG_DIR = r"C:\Projects\rust-rl-agent\ai-agent"

shards = [
    {"proc_id": 0, "bots": [0, 1], "cores": [1]},
    {"proc_id": 1, "bots": [2, 3], "cores": [2]},
    {"proc_id": 2, "bots": [4, 5], "cores": [3]},
    {"proc_id": 3, "bots": [6, 7], "cores": [4]},
]

processes = []

print("🚀 SYSTEM AFFINITY LOCKDOWN: 4-Shard / 2-Bot per Core Protocol 🚀")
print("Cores: 1, 2, 3, 4 active. Core 5 reserved for Server. Core 0 reserved for OS.")

for shard in shards:
    log_file = os.path.join(LOG_DIR, f"collection_shard_{shard['proc_id']}.log")
    cmd = [
        PYTHON_EXE, SCRIPT_PATH,
        "--proc_id", str(shard['proc_id']),
        "--bots"
    ] + [str(b) for b in shard['bots']] + ["--cpu_cores"] + [str(c) for c in shard['cores']]
    
    # Each shard gets its own log file via redirection
    with open(log_file, "w") as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=f, creationflags=subprocess.CREATE_NEW_CONSOLE)
        processes.append(p)
    print(f"Started Shard {shard['proc_id']} (Bots {shard['bots']}) -> {log_file}")
    time.sleep(5) # Stagger load to avoid VRAM/GIL contention

print("\nSwarm Active. Monitor logs/collection_shard_*.log for heartbeats.")
