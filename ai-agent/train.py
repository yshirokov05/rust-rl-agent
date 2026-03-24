import os
import time
import torch_directml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from environment import RustEnv
import json

# --- CONFIGURATION ---
os.environ["WANDB_MODE"] = "disabled"
CHECKPOINT_DIR = r"C:\Projects\rust-rl-agent\models\v2_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
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
        self.start_step = 0

    def _on_training_start(self) -> None:
        self.start_step = self.model.num_timesteps

    def _on_step(self) -> bool:
        if time.time() - self.last_sync > 30:
            elapsed = time.time() - self.start_time
            steps = self.model.num_timesteps - self.start_step
            sps = steps / elapsed if elapsed > 0 else 0
            
            stats = {
                "sps": sps,
                "live_step": self.model.num_timesteps,
                "device": "AMD_DirectML_5700XT",
                "active_agents": 1,
                "timestamp": time.time()
            }
            with open(STATS_PATH, 'w') as f:
                json.dump(stats, f)
            self.last_sync = time.time()
        return True

def train():
    device = torch_directml.device()
    print(f"DirectML Device: {device}", flush=True)

    env = DummyVecEnv([make_env(0)])
    env = VecMonitor(env)

    # Load Expert Model
    model_path = os.path.join(CHECKPOINT_DIR, "overnight_win_1591769_steps.zip")
    if os.path.exists(model_path):
        print(f"Loading Expert Model: {model_path}", flush=True)
        model = PPO.load(model_path, env=env, device=device)
    else:
        print("Expert model not found. Starting fresh.", flush=True)
        model = PPO("MlpPolicy", env, verbose=1, device=device, ent_coef=0.01)

    print("Training Started (WandB Disabled).", flush=True)
    model.learn(total_timesteps=10_000_000, callback=ProgressPulseCallback())

if __name__ == "__main__":
    train()
