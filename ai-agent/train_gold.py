import os
import time
import torch_directml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from environment import RustEnv
import json
import sys

# --- CONFIGURATION ---
os.environ["WANDB_MODE"] = "disabled"
ROOT_DIR = r"C:\Projects\rust-rl-agent"
CHECKPOINT_DIR = r"C:\Projects\rust-rl-agent\models\v2_checkpoints"
STATS_PATH = os.path.join(CHECKPOINT_DIR, "live_stats.json")

def make_env(rank, seed=0):
    def _init():
        env = RustEnv(rank)
        env = Monitor(env)
        return env
    return _init

class ProgressPulseCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.last_sync = time.time()
        self.start_time = time.time()
        self.initial_steps = 0

    def _on_training_start(self) -> None:
        self.initial_steps = self.model.num_timesteps
        print(f"BATTLE_LOG: Starting at step {self.initial_steps}", flush=True)

    def _on_step(self) -> bool:
        if time.time() - self.last_sync > 5:
            elapsed = time.time() - self.start_time
            # Total Absolute Steps = Initial + Steps in this run
            current_total = self.model.num_timesteps
            sps = (current_total - self.initial_steps) / elapsed if elapsed > 0 else 0
            
            stats = {
                "sps": sps,
                "live_step": current_total,
                "device": "AMD_DirectML_5700XT",
                "active_agents": 1,
                "timestamp": time.time()
            }
            try:
                with open(STATS_PATH, 'w') as f:
                    json.dump(stats, f)
            except: pass
            self.last_sync = time.time()
        return True

def train():
    device = torch_directml.device()
    print(f"DirectML Device: {device}", flush=True)

    env = DummyVecEnv([make_env(0)])
    env = VecMonitor(env)

    # LOAD THE LATEST VERIFIED CHECKPOINT (1.7M)
    model_path = os.path.join(CHECKPOINT_DIR, "overnight_win_1766769_steps.zip")
    
    if not os.path.exists(model_path):
        print(f"FATAL ERROR: Checkpoint {model_path} missing. Hallucination Guard engaged.", flush=True)
        sys.exit(1)

    print(f"Loading Expert Model: {model_path}", flush=True)
    model = PPO.load(model_path, env=env, device=device)
    
    print(f"Training Started (GOLD_PATH). Step: {model.num_timesteps}", flush=True)
    # CRITICAL: reset_num_timesteps=False to keep the 1.7M count
    model.learn(total_timesteps=10_000_000, callback=ProgressPulseCallback(), reset_num_timesteps=False)

if __name__ == "__main__":
    train()
