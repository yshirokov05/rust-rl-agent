import os
import time
import threading
import json
import traceback
import torch
import torch_directml
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
import wandb

# --- CONFIGURATION ---
CHECKPOINT_DIR = "models/v2_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def make_env(rank, seed=0):
    """
    Utility function for multiprocessed env.
    :param rank: (int) index of the subprocess
    """
    from environment import RustEnv
    def _init():
        env = RustEnv(bot_id=rank)
        env = Monitor(env)
        return env
    return _init

class SimpleLogCallback(BaseCallback):
    def __init__(self, save_freq, save_path, name_prefix="rl_model", verbose=1):
        super(SimpleLogCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _on_step(self) -> bool:
        # Aggregate logs from all parallel envs
        infos = self.locals.get("infos", [])
        if infos and self.num_timesteps % 50 == 0:
            total_wood = sum(info.get("wood_count", 0) for info in infos)
            total_cloth = sum(info.get("cloth_count", 0) for info in infos)
            
            wandb.log({
                "pulse/total_wood": total_wood,
                "pulse/total_cloth": total_cloth,
                "EMERGENCY_DASHBOARD/live_step": self.num_timesteps,
                "EMERGENCY_DASHBOARD/cloud_heartbeat": 1
            }, commit=True)

        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps")
            self.model.save(path)
        return True

def progress_pulse_thread(model_ref):
    """Background thread to log SPS and hardware state to W&B every 30s."""
    last_step = 0
    start_time = time.time()
    while True:
        time.sleep(30)
        try:
            if model_ref[0] is not None:
                current_step = model_ref[0].num_timesteps
                sps = (current_step - last_step) / 30.0 if last_step > 0 else 0
                last_step = current_step
                
                wandb.log({
                    "EMERGENCY_DASHBOARD/sps": sps,
                    "EMERGENCY_DASHBOARD/live_step": current_step,
                    "pulse/device_type": "AMD_DirectML_5700XT",
                    "pulse/active_agents": 8
                }, commit=True)
                print(f"[PULSE] Step: {current_step} | SPS: {sps:.2f} | Agents: 8", flush=True)
        except Exception as e:
            print(f"Pulse Error: {e}")

def train():
    num_envs = 8
    # Use SubprocVecEnv for true parallelism
    print(f"Initializing {num_envs} Parallel Environments...", flush=True)
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    env = VecMonitor(env)

    run = wandb.init(
        project="rust-rl-v2-modular",
        name="MultiBot_Extreme_Saturation",
        mode="online",
        resume="allow",
        id="deft-disco-1",
    )

    batch_size = 1024
    n_steps = 512 # 512 per env * 8 envs = 4096 rollout
    model_ref = [None]

    t_pulse = threading.Thread(target=progress_pulse_thread, args=(model_ref,), daemon=True)
    t_pulse.start()

    latest_model_path = os.path.join(CHECKPOINT_DIR, "latest_model.zip")
    dml_device = torch_directml.device()
    
    if os.path.exists(latest_model_path):
        print(f"Resuming Multi-Bot training from {latest_model_path}...")
        model = PPO.load(latest_model_path, env=env, device=dml_device)
        model.batch_size = batch_size
        model.n_steps = n_steps
        model.n_epochs = 30
    else:
        print("Starting new Multi-Bot training session...")
        model = PPO("MlpPolicy", env, verbose=1, device=dml_device, batch_size=batch_size, n_steps=n_steps, n_epochs=30, ent_coef=0.05)
    
    model_ref[0] = model
    callback = SimpleLogCallback(save_freq=5000, save_path=CHECKPOINT_DIR)

    try:
        print(f"Starting Multi-Bot training loop... Target: Hardware Saturation", flush=True)
        model.learn(total_timesteps=10000000, callback=callback, reset_num_timesteps=False)
    except KeyboardInterrupt:
        print("Training interrupted.")
    except Exception as e:
        print(f"Training Crash: {e}")
        traceback.print_exc()
    finally:
        model.save(latest_model_path)
        wandb.finish()

if __name__ == "__main__":
    train()
