import json
import argparse
import random
import subprocess
import os
import time
import orjson
import psutil
import torch
import torch_directml
import numpy as np
import threading
import queue
from pathlib import Path
from stable_baselines3 import PPO

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)

class AsyncLogger:
    def __init__(self):
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
    def log(self, msg): self.queue.put(msg)
    def _run(self):
        while True:
            msg = self.queue.get()
            print(msg, flush=True)

logger = AsyncLogger()

# --- CONFIGURATION ---
PID = None  # Resolved at runtime by scanning for RustDedicated.exe
MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", "v2_checkpoints", "final_mastery_model.zip")
VISION_PATH = os.path.join(_PROJECT_ROOT, "shared-data", "vision_0.json")
PROJECT_STATE_PATH = os.path.join(_PROJECT_ROOT, "project_state.json")
LOOP_INTERVAL = 0.02  # 50Hz (20ms)
RAM_THRESHOLD_GB = 2.5
GPU_THRESHOLD_PERCENT = 5.0
DEPTH_LOG_DIR = os.path.join(_PROJECT_ROOT, "shared-data", "depth_logs")
os.makedirs(DEPTH_LOG_DIR, exist_ok=True)

def update_state(status, notes):
    import json
    try:
        with open(PROJECT_STATE_PATH, 'r') as f:
            state = json.load(f)
        state['status'] = status
        state['notes'] = notes
        with open(PROJECT_STATE_PATH, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"Error updating project_state.json: {e}")

FILE_HANDLES = {}

def get_obs(bot_id):
    path = os.path.join(_PROJECT_ROOT, "shared-data", f"vision_{bot_id}.json")
    if bot_id not in FILE_HANDLES:
        if not os.path.exists(path):
            return None, None, False
        try:
            FILE_HANDLES[bot_id] = open(path, 'rb')
        except:
            return None, None, False

    f = FILE_HANDLES[bot_id]
    try:
        f.seek(0)
        data = orjson.loads(f.read())

        player = data.get('PlayerPosition') or {'X': 0, 'Y': 0, 'Z': 0}
        tree_data = data.get('NearestTree') or {}
        tree = tree_data.get('Position') or {'X': 0, 'Y': 0, 'Z': 0}
        ore_data = data.get('NearestOre') or {}
        ore = ore_data.get('Position') or {'X': 0, 'Y': 0, 'Z': 0}
        health = data.get('Health', 100.0)
        wood = data.get('WoodCount', 0)
        stone = data.get('StoneCount', 0)
        predator = 1.0 if data.get('IsPredatorNearby', False) else 0.0
        active_item = data.get('ActiveItem', 'none')
        item_id = 0
        if 'plan' in active_item: item_id = 1
        elif 'hammer' in active_item: item_id = 2
        elif 'rock' in active_item: item_id = 3

        vec_obs = np.array([
            player['X'] / 1000.0, player['Y'] / 1000.0, player['Z'] / 1000.0,
            tree['X'] / 100.0, tree['Y'] / 100.0, tree['Z'] / 100.0,
            ore['X'] / 100.0, ore['Y'] / 100.0, ore['Z'] / 100.0,
            health / 100.0,
            wood / 1000.0,
            stone / 1000.0,
            predator,
            item_id / 3.0
        ], dtype=np.float32)

        # Build semantic image from SemanticMapBase64 if present, else zeros
        b64_map = data.get('SemanticMapBase64', '')
        if b64_map:
            import base64
            import cv2
            raw_bytes = base64.b64decode(b64_map)
            map_array = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((84, 84, 3))
            map_array = cv2.resize(map_array, (224, 224), interpolation=cv2.INTER_NEAREST)
            image_obs = np.transpose(map_array, (2, 0, 1))
        else:
            image_obs = np.zeros((3, 224, 224), dtype=np.uint8)

        obs = {"image": image_obs, "vector": vec_obs}
        return obs, data.get('DepthMatrix'), data.get('HasGathered', False)
    except:
        return None, None, False

TELEPORT_QUEUE = {} # bot_id: [x, y, z]

def write_actions(bot_id, action):
    import json
    # action: [Forward, Strafe, LookX, LookY, Sprint, Jump, Attack]
    # BotController expects: { Forward, Strafe, Jump, Attack, Sprint }
    path = os.path.join(_PROJECT_ROOT, "shared-data", f"actions_{bot_id}.json")
    data = {
        "Forward": float(action[0]),
        "Strafe": float(action[1]),
        "Jump": float(action[5]),
        "Attack": float(action[6]),
        "Sprint": float(action[4]),
        "Respawn": False
    }
    
    # PERSISTENT COMMANDS (Stay for 50 ticks to ensure plugin reads them)
    if bot_id in TELEPORT_QUEUE:
        if "counter" not in TELEPORT_QUEUE[bot_id]:
            TELEPORT_QUEUE[bot_id] = {"cmd": TELEPORT_QUEUE[bot_id], "counter": 50}
        
        cmd_info = TELEPORT_QUEUE[bot_id]
        pos = cmd_info["cmd"]
        if pos == "RESPAWN":
            data["Respawn"] = True
        else:
            data["TeleportPos"] = {"X": pos[0], "Y": pos[1], "Z": pos[2]}
        
        cmd_info["counter"] -= 1
        if cmd_info["counter"] <= 0:
            TELEPORT_QUEUE.pop(bot_id)

    try:
        with open(path, 'w') as f:
            json.dump(data, f)
    except:
        pass

def run_inference(proc_id, bots, cpu_cores=None):
    global PID
    if PID is None:
        PID = 0  # Will be resolved below
    
    # CPU Affinity Pinning
    if cpu_cores:
        try:
            p = psutil.Process()
            p.cpu_affinity(cpu_cores)
            print(f"[Proc {proc_id}] CPU Affinity Pinned to cores: {cpu_cores}")
        except Exception as e:
            print(f"[Proc {proc_id}] Affinity Error: {e}")

    SHARD_LOG_DIR = os.path.join(DEPTH_LOG_DIR, f"proc_{proc_id}")
    os.makedirs(SHARD_LOG_DIR, exist_ok=True)
    
    print(f"[Proc {proc_id}] Searching for RustDedicated.exe...")
    found = False
    for proc in psutil.process_iter(['name', 'pid']):
        if proc.info['name'] == "RustDedicated.exe":
            PID = proc.info['pid']
            found = True
            break
    
    if not found:
        print(f"[Proc {proc_id}] CRITICAL: RustDedicated.exe not found.")
        return

    print(f"[Proc {proc_id}] Binding to PID {PID} for Bots {bots}...")
    rust_proc = psutil.Process(PID)

    device = torch.device("cpu")
    if torch_directml.is_available():
        device = torch_directml.device()
    
    print(f"[Proc {proc_id}] Loading Model on {device}...", flush=True)
    model = PPO.load(MODEL_PATH, device=device)

    # Update state only once at start to avoid flickering
    if proc_id == 0:
        update_state("Inference Active", f"Swarm Scaling Active. GPU Shards Running.")
    
    batch_buffer = []
    batch_idx = 0
    step_counter = 0
    active_bot_count = len(bots)
    last_heartbeat = time.time()
    
    bot_stats = {bid: {"last_pos": None, "last_move": time.time()} for bid in bots}

    logger.log(f"[Proc {proc_id}] ELITE VANGUARD: 8-Bot / 50Hz Overclock Active.")
    print(f"[Proc {proc_id}] STARTING INFERENCE LOOP", flush=True)

    loop_times = []

    try:
        while True:
            if not rust_proc.is_running():
                break

            # RAM SAFETY WATCHDOG
            mem = psutil.virtual_memory().percent
            if mem > 95.0:
                logger.log(f"[Proc {proc_id}] CRITICAL: RAM at {mem}%. Emergency shutdown.")
                os._exit(1)

            if time.time() - last_heartbeat > 60:
                logger.log(f"[Proc {proc_id}] Still Alive: {batch_idx * 200} steps collected.")
                last_heartbeat = time.time()

            if step_counter % 100 == 0:
                try: temp = get_gpu_temp()
                except: temp = 45.0
                if temp > 85.0:
                    active_bot_count = max(1, len(bots) // 2)
                    print(f"[Proc {proc_id}] THERMAL THROTTLE: {temp}C > 85C. Dropping to half capacity.")
                else:
                    active_bot_count = len(bots)

            for i in range(active_bot_count):
                bot_id = bots[i]
                obs, depth_matrix, has_gathered = get_obs(bot_id)
                
                if obs is not None:
                    # Stuck Detection
                    current_pos = obs["vector"][0:3]  # Player X,Y,Z (normalized)
                    stats = bot_stats[bot_id]
                    
                    # STUCK DETECTION (10s Threshold for Jitter)
                    if stats["last_pos"] is not None:
                        dist = np.linalg.norm(current_pos - stats["last_pos"])
                        if dist < 0.1: # Bot hasn't moved significantly
                            if time.time() - stats["last_move"] > 10.0: # Stuck for 10 seconds
                                logger.log(f"[Proc {proc_id}] Bot_{bot_id} STALLED. Triggering 50-unit JITTER.")
                                # Denormalize from /1000 scale back to world coords
                                jx = current_pos[0] * 1000.0 + random.uniform(-50, 50)
                                jz = current_pos[2] * 1000.0 + random.uniform(-50, 50)
                                jy = current_pos[1] * 1000.0 + 2.0
                                TELEPORT_QUEUE[bot_id] = [jx, jy, jz]
                                stats["last_move"] = time.time() # Reset timer
                        else: # Bot moved
                            stats["last_pos"] = current_pos
                            stats["last_move"] = time.time()
                    else: # First observation for this bot
                        stats["last_pos"] = current_pos
                        stats["last_move"] = time.time()

                if obs is not None and depth_matrix is not None:
                    action, _ = model.predict(obs, deterministic=True)
                    
                    depth_arr = np.array(depth_matrix, dtype=np.float32).reshape(84, 84)
                    # STRICT NORMALIZATION (0-1)
                    depth_arr = np.clip(depth_arr / 100.0, 0, 1)

                    batch_buffer.append({
                        "obs": obs,
                        "action": action,
                        "reward": float(has_gathered),
                        "depth": depth_arr
                    })
                    
                    if len(batch_buffer) >= 200:
                        ts = int(time.time() * 1000)
                        filename = os.path.join(SHARD_LOG_DIR, f"batch_{batch_idx}_{ts}.npz")
                        np.savez(
                            filename,
                            obs=np.array([s["obs"] for s in batch_buffer]),
                            action=np.array([s["action"] for s in batch_buffer]),
                            reward=np.array([s["reward"] for s in batch_buffer]),
                            depth=np.array([s["depth"] for s in batch_buffer])
                        )
                        batch_buffer = []
                        batch_idx += 1
                        # update_state("Inference Active", f"Proc {proc_id} Batched {batch_idx*200} steps.")

                    write_actions(bot_id, action)

            step_counter += 1
            
            # SPS Calculation
            loop_times.append(time.time())
            if len(loop_times) > 100:
                loop_times.pop(0)
                sps = len(loop_times) / (loop_times[-1] - loop_times[0])
                if step_counter % 500 == 0:
                    logger.log(f"[Proc {proc_id}] CURRENT SPS: {sps * active_bot_count:.2f} (Combined)")

            # Precise Timing for 50Hz
            time.sleep(LOOP_INTERVAL)

    except Exception as e:
        print(f"[Proc {proc_id}] CRITICAL ERROR: {e}")

def get_dir_size(path):
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
    return total / (1024 ** 3) # GB

def get_gpu_temp():
    try:
        # Fallback to a common WMI path for thermal zones
        cmd = "powershell -Command \"(Get-CimInstance -ClassName Win32_PerfFormattedData_GPUPerformanceCounters_GPUDevice -ErrorAction SilentlyContinue).Temperature\""
        output = subprocess.check_output(cmd, shell=True).decode().strip()
        if output:
            return float(output)
    except:
        pass
    return 45.0 # Safe default if unreadable


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--proc_id", type=int, default=0)
    parser.add_argument("--bots", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--cpu_cores", nargs="+", type=int, default=None)
    args = parser.parse_args()
    
    # Inject Bot_0 Recovery Plan
    if 0 in args.bots:
        # First action: Respawn, Second: Teleport to Bot 4's Metal Zone
        TELEPORT_QUEUE[0] = "RESPAWN"
        def inject_tele():
            time.sleep(1.0)
            TELEPORT_QUEUE[0] = [467.5, 6.9, 949.0]
        threading.Thread(target=inject_tele, daemon=True).start()
        
    run_inference(args.proc_id, args.bots, args.cpu_cores)
