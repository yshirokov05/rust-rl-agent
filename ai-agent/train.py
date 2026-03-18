import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from wandb.integration.sb3 import WandbCallback
import wandb
from environment import RustEnv
import os
import torch

from stable_baselines3.common.monitor import Monitor

def train():
    # Paths
    vision_path = "shared-data/vision.json"
    models_dir = "models"
    checkpoints_dir = os.path.join(models_dir, "checkpoints")
    
    # Ensure directories exist
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(os.path.dirname(vision_path), exist_ok=True)
    
    # Create the environment and wrap with Monitor
    env = RustEnv(vision_path=vision_path)
    env = Monitor(env)
    
    # Initialize WandB - DISABLED for stability
    # run = wandb.init(
    #     project="rust-rl-agent",
    #     mode="offline",
    #     sync_tensorboard=True,
    #     monitor_gym=True,
    #     save_code=True,
    # )

    # Resume Logic (Check for latest_model.zip)
    latest_model_path = os.path.join(models_dir, "latest_model.zip")
    if os.path.exists(latest_model_path):
        print(f"Resuming training from {latest_model_path}")
        model = PPO.load(latest_model_path, env=env, device="auto")
    else:
        print("Starting new training session")
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="runs/v1_2", device="auto")
    
    class SimpleLogCallback(CheckpointCallback):
        def _on_step(self) -> bool:
            if self.n_calls % 100 == 0:
                infos = self.locals["infos"][0]
                goal = infos.get("current_goal", "Thinking...")
                reward = infos.get("reward", 0.0)
                steps = self.num_timesteps
                print(f"[{time.ctime()}] Groundbreaker: {goal} | Reward: {reward:.1f} | Total Steps: {steps}")
            return super()._on_step()

    simple_callback = SimpleLogCallback(
        save_freq=5000,
        save_path=checkpoints_dir,
        name_prefix="rust_rl_model",
    )

    print("Starting training... (Wait for plugin to generate data)")
    
    # Train the agent
    try:
        model.learn(
            total_timesteps=1000000, 
            callback=[simple_callback],
            progress_bar=False
        )
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        # Save the final model as 'latest_model' for future resumes
        model.save(latest_model_path)
        print(f"Model saved to {latest_model_path}")
        wandb.finish()

if __name__ == "__main__":
    train()
