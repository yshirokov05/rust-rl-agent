import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from wandb.integration.sb3 import WandbCallback
import wandb
from environment import RustEnv
import os
import torch

def train():
    # Paths
    vision_path = "shared-data/vision.json"
    models_dir = "models"
    checkpoints_dir = os.path.join(models_dir, "checkpoints")
    latest_model_path = os.path.join(models_dir, "latest.zip")
    
    # Ensure directories exist
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(os.path.dirname(vision_path), exist_ok=True)
    
    # Create the environment
    env = RustEnv(vision_path=vision_path)
    
    # Initialize WandB
    run = wandb.init(
        project="rust-agent",
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )

    # Resume Logic
    if os.path.exists(latest_model_path):
        print(f"Resuming training from {latest_model_path}")
        model = PPO.load(latest_model_path, env=env, device="auto")
    else:
        print("Starting new training session")
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=f"runs/{run.id}", device="auto")
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=checkpoints_dir,
        name_prefix="rust_rl_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    wandb_callback = WandbCallback(
        gradient_save_freq=100,
        model_save_path=os.path.join(models_dir, f"run_{run.id}"),
        verbose=2,
    )

    print("Starting training... (Wait for plugin to generate data)")
    
    # Train the agent
    # Using a high number for total_timesteps as we rely on checkpoints/manual stop
    try:
        model.learn(
            total_timesteps=1000000, 
            callback=[checkpoint_callback, wandb_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        # Save the final model as 'latest' for future resumes
        model.save(latest_model_path)
        print(f"Model saved to {latest_model_path}")
        wandb.finish()

if __name__ == "__main__":
    train()
