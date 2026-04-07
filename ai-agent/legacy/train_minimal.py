import os
import time
import torch_directml
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from environment import RustEnv
import wandb
from wandb.integration.sb3 import WandbCallback

# --- CONFIGURATION ---
os.environ["WANDB_CONSOLE"] = "off"
os.environ["WANDB_MODE"] = "offline"
WANDB_PROJECT = "rust-rl-v2-modular"
WANDB_RUN_ID = "deft-disco-1"
CHECKPOINT_DIR = r"C:\Projects\rust-rl-agent\models\v2_checkpoints"
EXPERT_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "overnight_win_1766769_steps.zip")

class DiversificationCallback(BaseCallback):
    """
    Handles Dynamic Teleport every 250k steps and Entropy Watch for LR shakes.
    """
    def __init__(self, verbose=0):
        super(DiversificationCallback, self).__init__(verbose)
        self.last_teleport_step = 0
        self.teleport_interval = 250000
        self.entropy_threshold = 0.05
        self.shake_duration = 5000
        self.in_shake = False
        self.shake_start_step = 0
        self.original_lr = 0.0003 # Default PPO
        self.shaked_lr = 0.001

    def _on_step(self) -> bool:
        try:
            current_step = self.model.num_timesteps
            
            # 1. DYNAMIC TELEPORT
            if current_step - self.last_teleport_step >= self.teleport_interval:
                self.last_teleport_step = current_step
                self._trigger_teleport()

            # 2. ENTROPY WATCH
            # Access entropy via logger safely
            if "train/entropy" in self.logger.name_to_value:
                entropy = self.logger.name_to_value["train/entropy"]
                if entropy < self.entropy_threshold and not self.in_shake:
                    print(f">>> ENTROPY CRITICAL ({entropy:.4f}). TRIGGERING LR SHAKE.")
                    self.in_shake = True
                    self.shake_start_step = current_step
                    self.model.learning_rate = self.shaked_lr
                
            # End shake after duration
            if self.in_shake and (current_step - self.shake_start_step >= self.shake_duration):
                print(f">>> SHAKE COMPLETE. RESTORING NORMAL LR.")
                self.in_shake = False
                self.model.learning_rate = self.original_lr
        except Exception as e:
            print(f"Callback Error: {e}")
            
        return True

    def _trigger_teleport(self):
        print(">>> TRIGGERING DYNAMIC TELEPORT (ANTI-MEMORIZATION)")
        # Center of Palm Forest (approx)
        center_x, center_z = 1200, -800
        radius = 500
        
        # Random point in circle
        angle = np.random.uniform(0, 2 * np.pi)
        r = radius * np.sqrt(np.random.uniform(0, 1))
        new_x = center_x + r * np.cos(angle)
        new_z = center_z + r * np.sin(angle)
        
        # Command server to teleport bot
        teleport_cmd = f"teleportpos {new_x} 10 {new_z}"
        print(f"DEBUG: Executing {teleport_cmd}")
        # In a real setup, we'd write this to a RCON command file or similar.
        # For now, we log it; the BotController plugin should handle the actual move if tied to a file.
        with open("shared-data/server_cmds.json", "w") as f:
            import json
            json.dump({"command": teleport_cmd}, f)

def make_env():
    env = RustEnv(bot_id=0)
    env = Monitor(env)
    return env

def train():
    device = torch_directml.device()
    env = DummyVecEnv([make_env])
    env = VecMonitor(env)

    if os.path.exists(EXPERT_MODEL_PATH):
        model = PPO.load(EXPERT_MODEL_PATH, env=env, device=device)
        print(f"VERIFIED STARTING STEP: {model.num_timesteps}")
    else:
        print("ABORTING: No expert model.")
        return

    run = wandb.init(
        project=WANDB_PROJECT,
        id=WANDB_RUN_ID,
        resume="must",
        sync_tensorboard=True,
        settings=wandb.Settings(start_method="spawn")
    )
    
    wandb.define_metric("*", step_metric="time/total_timesteps")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=25000,
        save_path=CHECKPOINT_DIR,
        name_prefix="overnight_win"
    )
    
    wandb_callback = WandbCallback(gradient_save_freq=0, verbose=2)
    div_callback = DiversificationCallback()

    print(">>> COMMENCING 4,000,000 STEP MASTERY RUN")
    try:
        model.learn(
            total_timesteps=4000000 - model.num_timesteps,
            callback=[checkpoint_callback, wandb_callback, div_callback],
            reset_num_timesteps=False,
            progress_bar=True
        )
    except KeyboardInterrupt:
        pass
    finally:
        model.save(os.path.join(CHECKPOINT_DIR, "final_mastery_model"))
        run.finish()

if __name__ == "__main__":
    train()
