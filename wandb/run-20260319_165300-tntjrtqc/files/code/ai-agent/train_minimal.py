import os
import time
import torch_directml
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
from environment import RustEnv
import wandb
from wandb.integration.sb3 import WandbCallback

# --- CONFIGURATION ---
WANDB_PROJECT = "rust-rl-agent"
WANDB_RUN_NAME = "Expert_Restoration_500k_Run"
CHECKPOINT_DIR = r"C:\Projects\rust-rl-agent\models\v2_checkpoints"
EXPERT_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "overnight_win_1766769_steps.zip")

def make_env():
    env = RustEnv(bot_id=0)
    env = Monitor(env)
    return env

def train():
    # 1. Initialize DirectML Device
    device = torch_directml.device()
    print(f"DEBUG: Active Device: {device}")

    # 2. Setup Environment
    env = DummyVecEnv([make_env])
    env = VecMonitor(env)

    # 3. Load Expert Model (with verification check)
    if os.path.exists(EXPERT_MODEL_PATH):
        print(f"LOADING EXPERT BRAIN: {EXPERT_MODEL_PATH}")
        # Note: reset_num_timesteps=False is set in model.learn()
        model = PPO.load(EXPERT_MODEL_PATH, env=env, device=device)
        print(f"VERIFIED STARTING STEP: {model.num_timesteps}")
    else:
        print("CRITICAL ERROR: Expert model not found. Aborting.")
        return

    # 4. Initialize W&B (Minimalist)
    run = wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=True,
    )

    # 5. Callbacks
    # Save every 25k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=25000,
        save_path=CHECKPOINT_DIR,
        name_prefix="overnight_win"
    )
    
    # W&B Logging (Minimal frequency to save CPU)
    wandb_callback = WandbCallback(
        gradient_save_freq=0,
        model_save_path=None,
        verbose=2
    )

    # 6. COMMENCE TRAINING
    print(">>> COMMENCING 500,000 STEP PRODUCTION RUN")
    try:
        model.learn(
            total_timesteps=500000,
            callback=[checkpoint_callback, wandb_callback],
            reset_num_timesteps=False, # CRITICAL: Maintain 1.76M count
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        model.save(os.path.join(CHECKPOINT_DIR, "final_restored_model"))
        run.finish()

if __name__ == "__main__":
    train()
