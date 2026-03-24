import gymnasium as gym
import time
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from wandb.integration.sb3 import WandbCallback
import wandb
from environment import RustEnv
from environment import RustEnv
import os
import torch
import matplotlib.pyplot as plt
import io
import matplotlib
matplotlib.use('Agg')

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
    
    # Initialize WandB
    run = wandb.init(
        project="rust-rl-v2-modular",
        mode="offline",
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )

    # Resume Logic (Check for latest_model.zip)
    latest_model_path = os.path.join(models_dir, "latest_model.zip")
    if os.path.exists(latest_model_path):
        print(f"Resuming training from {latest_model_path}")
        model = PPO.load(latest_model_path, env=env, device="auto")
        # Ensure model age is preserved after migration
        if model.num_timesteps < 147000:
            model.num_timesteps = 147000
    else:
        print("Starting new training session")
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="runs/v1_2", device="auto")
    
    class SimpleLogCallback(CheckpointCallback):
        def _on_step(self) -> bool:
            infos = self.locals["infos"][0]
            
            # Milestone alert
            cloth_count = infos.get("cloth_count", 0)
            if not hasattr(self, "milestone_achieved") and cloth_count >= 1:
                print("\n*** MILESTONE ACHIEVED: 1x Cloth Collected! ***\n", flush=True)
                self.milestone_achieved = True
                
            if self.n_calls % 100 == 0:
                goal = infos.get("current_goal", "Thinking...")
                reward = infos.get("reward", 0.0)
                steps = self.num_timesteps
                
                wood_count = infos.get("wood_count", 0)
                
                # Fetch learning metrics
                logger = self.model.logger.name_to_value
                kl = logger.get("train/approx_kl", 0.0)
                ent = logger.get("train/entropy_loss", 0.0)
                
                print(f"[{time.ctime()}] Groundbreaker: {goal} | Reward: {reward:.1f} | Steps: {steps} | KL: {kl:.5f} | Ent: {ent:.2f} | Wood: {wood_count} | Cloth: {cloth_count}", flush=True)
                
                # Render 2D Map
                obs = self.locals["new_obs"][0]
                px, pz = obs[0], obs[2]
                tx, tz = obs[3], obs[5]
                ox, oz = obs[6], obs[8]
                
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.set_facecolor('#1e1e1e')
                fig.patch.set_facecolor('#121212')
                
                # Plot Targets
                ax.scatter(tx, tz, color='green', s=100, label='Tree/Hemp', marker='^')
                ax.scatter(ox, oz, color='saddlebrown', s=100, label='Ore', marker='s')
                
                # Plot Agent
                ax.scatter(px, pz, color='dodgerblue', s=150, label='Agent')
                
                # Raycast Line (To Nearest Target)
                tree_dist = infos.get("tree_dist", 999.0)
                ore_dist = infos.get("ore_dist", 999.0)
                if tree_dist < ore_dist:
                    ax.plot([px, tx], [pz, tz], color='lime', linestyle='--', alpha=0.5)
                else:
                    ax.plot([px, ox], [pz, oz], color='orange', linestyle='--', alpha=0.5)
                
                ax.set_title(f"Live Feed - Step {steps}\nGoal: {goal}", color='white')
                ax.tick_params(colors='gray')
                ax.legend(loc='upper right', facecolor='#2a2a2a', edgecolor='none', labelcolor='white')
                ax.grid(color='gray', linestyle=':', alpha=0.3)
                
                # Equal aspect ratio but auto bounds
                ax.autoscale()
                ax.margins(0.2)
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                plt.close(fig)
                buf.seek(0)
                img = wandb.Image(buf)
                
                wandb.log({
                    "live_map": img,
                    "inventory/wood": wood_count,
                    "inventory/cloth": cloth_count,
                    "reward": reward
                }, step=steps)

            return super()._on_step()

    simple_callback = SimpleLogCallback(
        save_freq=2500,
        save_path=os.path.join("models", "v2_modular"),
        name_prefix="rust_rl_model",
    )

    print("Starting training... (Wait for plugin to generate data)")
    
    # Train the agent
    try:
        model.learn(
            total_timesteps=1000000, 
            callback=[simple_callback],
            progress_bar=False,
            reset_num_timesteps=False
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
