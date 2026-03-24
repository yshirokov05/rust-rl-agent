import os
import time
import torch_directml
import numpy as np
import json
from stable_baselines3 import PPO
from environment import RustEnv
from stable_baselines3.common.vec_env import DummyVecEnv

CHECKPOINT_DIR = r"C:\Projects\rust-rl-agent\models\v2_checkpoints"
# Found true Gold Model at 3.56M steps
FINAL_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "overnight_win_3566769_steps.zip")

def graduation_exam():
    print(f"LOADING FINAL MASTERY MODEL: {FINAL_MODEL_PATH}")
    device = torch_directml.device()
    env = DummyVecEnv([lambda: RustEnv(bot_id=0)])
    
    model = PPO.load(FINAL_MODEL_PATH, env=env, device=device)
    print(f"MODEL STEPS: {model.num_timesteps}")
    
    # DIVERSIFICATION TEST: Teleport into the general Forest Zone
    new_x, new_z = 800, -800
    teleport_cmd = f"teleportpos {new_x} 10 {new_z}"
    print(f"EXAM TELEPORT: {teleport_cmd}")
    with open("shared-data/server_cmds.json", "w") as f:
        json.dump({"command": teleport_cmd}, f)
    
    time.sleep(5) # Give more time for heavy zone load
    
    print("STARTING DETERMINISTIC TEST (n=1 episode)")
    obs = env.reset()
    start_time = time.time()
    hit_time = None
    total_swings = 0
    total_hits = 0
    
    # Run for 2000 steps
    for i in range(2000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        # MONITOR DETECTION
        tree_dist = info[0].get("tree_dist", 999)
        tree_name = info[0].get("current_goal", "")
        
        if tree_dist < 50:
             print(f"Step {i}: Vision Detected '{tree_name}' at {tree_dist:.2f}m")

        # Track hits
        if action[0][6] > 0: # Attack bit
            total_swings += 1
            if info[0].get("has_gathered"):
                total_hits += 1
                if hit_time is None:
                    hit_time = time.time() - start_time
                    print(f"FIRST HIT AT: {hit_time:.2f}s (Step {i})")
        
        if total_hits >= 5: 
            break
            
    efficiency = (total_hits / total_swings) if total_swings > 0 else 0
    print("-" * 30)
    print(f"GRADUATION TEST COMPLETE")
    print(f"Test Result: {'PASS' if total_hits >= 1 else 'FAIL'}")
    print(f"Time to first hit: {hit_time if hit_time else 'N/A'}s")
    print(f"Harvest Efficiency: {efficiency:.2%}")
    print("-" * 30)

if __name__ == "__main__":
    graduation_exam()
