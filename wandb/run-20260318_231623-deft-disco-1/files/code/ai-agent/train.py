import gymnasium as gym
import time
import os
import io
import json
import numpy as np
import threading
import torch
import torch_directml
import traceback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import wandb
from environment import RustEnv

VISION_PATH = "shared-data/vision.json"
ACTIONS_PATH = "shared-data/actions.json"

def safe_load_json(path):
    try:
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
    except:
        pass
    return {}

def background_dashboard_thread():
    """Generates the 2D matplotlib map and logs it to W&B every 2 seconds without blocking training."""
    while True:
        time.sleep(2)
        try:
            vision_data = safe_load_json(VISION_PATH)
            action_data = safe_load_json(ACTIONS_PATH)
            if not vision_data or not action_data:
                continue
                
            px = vision_data.get("PlayerPosition", {}).get("X", 0)
            pz = vision_data.get("PlayerPosition", {}).get("Z", 0)
            
            tx = vision_data.get("NearestTree", {}).get("Position", {}).get("X", 0)
            tz = vision_data.get("NearestTree", {}).get("Position", {}).get("Z", 0)
            
            ox = vision_data.get("NearestOre", {}).get("Position", {}).get("X", 0)
            oz = vision_data.get("NearestOre", {}).get("Position", {}).get("Z", 0)
            
            is_harvesting = action_data.get("Attack", False)
            
            fig, ax = plt.subplots(figsize=(6, 6))
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#0e1117')
            
            dist_tree = np.linalg.norm([px - tx, pz - tz])
            dist_ore = np.linalg.norm([px - ox, pz - oz])
            nearest_x, nearest_z = (tx, tz) if dist_tree < dist_ore else (ox, oz)
            
            tree_color = 'yellow' if (is_harvesting and dist_tree < dist_ore) else 'green'
            ore_color = 'yellow' if (is_harvesting and dist_tree >= dist_ore) else 'saddlebrown'
            
            if is_harvesting:
                if dist_tree < dist_ore:
                    ax.scatter(tx, tz, color=tree_color, s=500, alpha=0.5)
                else:
                    ax.scatter(ox, oz, color=ore_color, s=500, alpha=0.5)
                    
            ax.scatter(px, pz, color='dodgerblue', s=200, label='Agent')
            ax.scatter(tx, tz, color=tree_color, s=150, marker='^', label='Tree/Hemp')
            ax.scatter(ox, oz, color=ore_color, s=150, marker='s', label='Ore')
            ax.plot([px, nearest_x], [pz, nearest_z], color='lime' if is_harvesting else 'white', linestyle='--', alpha=0.5)
            
            all_x = [px, tx, ox]
            all_z = [pz, tz, oz]
            ax.set_xlim(min(all_x) - 5.0, max(all_x) + 5.0)
            ax.set_ylim(min(all_z) - 5.0, max(all_z) + 5.0)
            
            ax.legend(facecolor='#1e1e1e', edgecolor='none', labelcolor='white', loc='upper left')
            ax.grid(color='gray', linestyle=':', alpha=0.2)
            ax.set_title("Raycast & Harvest Targeting (W&B Live)", color='white')
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            img = wandb.Image(buf)
            
            wandb.log({"live_map": img}, commit=False)
        except Exception as e:
            pass

def progress_pulse_thread(model_ref, env_ref):
    """Logs progress every 30 seconds and stops everything if stalled for 2 minutes."""
    last_step = -1
    last_time = time.time()
    stall_timer = 0
    
    while True:
        time.sleep(30)
        try:
            model = model_ref[0] # Using array ref to get latest model
            if model is None:
                continue
            
            current_step = model.num_timesteps
            
            # Step Throughput (SPS) Tracking
            current_time = time.time()
            elapsed_time = current_time - last_time
            sps = (current_step - last_step) / elapsed_time if last_step >= 0 and elapsed_time > 0 else 0.0
            last_time = current_time
            
            wandb.log({"pulse/sps": sps}, commit=False)
            
            # Relying on native W&B API telemetry for AMD GPU Load instead
            gpu_load = "W&B (AMD)"
                
            # Get latest reward from the SB3 logger if possible
            last_reward = 0.0
            if model.logger and model.logger.name_to_value:
                last_reward = model.logger.name_to_value.get("rollout/ep_rew_mean", 0.0)
                
            print(f"Status: Training Active. Step: {current_step}. Last Reward: {last_reward:.1f}. GPU Load: {gpu_load}", flush=True)
            
            if current_step == last_step:
                stall_timer += 30
                if stall_timer >= 120:
                    print(f"ERROR: Step count has not moved from {current_step} for 2 minutes. Stalled. Stopping everything.", flush=True)
                    os._exit(1)
            else:
                last_step = current_step
                stall_timer = 0
        except Exception as e:
            pass

class SimpleLogCallback(CheckpointCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.harvest_history = []

    def _on_step(self) -> bool:
        infos = self.locals["infos"][0]
        cloth_count = infos.get("cloth_count", 0)
        wood_count = infos.get("wood_count", 0)
        reward = infos.get("reward", 0.0)
        is_harvesting = infos.get("is_harvesting", 0)
        
        self.harvest_history.append(is_harvesting)
        if len(self.harvest_history) > 100:
            self.harvest_history.pop(0)
        
        if not hasattr(self, "milestone_achieved") and cloth_count >= 1:
            print("\n*** MILESTONE ACHIEVED: 1x Cloth Collected! ***\n", flush=True)
            self.milestone_achieved = True
            
        if self.n_calls == 1000 and hasattr(self, "model") and hasattr(self.model.policy, "optimizer"):
            for param_group in self.model.policy.optimizer.param_groups:
                if param_group['lr'] == 1e-5:
                    param_group['lr'] = 3e-4
                    print(f"\n*** SAFETY BUFFER COMPLETE: Restored learning rate to 3e-4 at total step {self.num_timesteps} ***\n", flush=True)
            
        if self.n_calls % 100 == 0:
            harvest_rate = sum(self.harvest_history) / len(self.harvest_history) if self.harvest_history else 0.0
            wandb.log({
                "inventory/wood": wood_count,
                "inventory/cloth": cloth_count,
                "reward_step": reward,
                "action_metrics/harvest_rate": harvest_rate * 100.0
            }, step=self.num_timesteps)
        return super()._on_step()

def train():
    models_dir = "models"
    checkpoints_dir = os.path.join(models_dir, "v2_checkpoints")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(os.path.dirname(VISION_PATH), exist_ok=True)
    
    run = wandb.init(
        project="rust-rl-v2-modular",
        mode="online", # Synced to W&B cloud dashboard
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
        resume="allow",
        id="deft-disco-1",
    )
    
    env = Monitor(RustEnv(vision_path=VISION_PATH))
    
    # Independent Evaluation Validation Tracker
    eval_env = Monitor(RustEnv(vision_path=VISION_PATH))
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(checkpoints_dir, "best_model"),
        log_path=checkpoints_dir,
        eval_freq=10000,
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )
    
    latest_model_path = os.path.join(checkpoints_dir, "latest_model.zip")
    
    batch_size = 64 # Default
    model_ref = [None] # Array ref so pulse thread can see the model
    
    # Start background W&B visualizer
    t_dash = threading.Thread(target=background_dashboard_thread, daemon=True)
    # t_dash.start() # Temporarily disabled to prevent GIL locking the training loop

    
    # Start pulse logger
    t_pulse = threading.Thread(target=progress_pulse_thread, args=(model_ref, env), daemon=True)
    t_pulse.start()
    
    while True: # Loop for OOM Auto-Restart
        try:
            pre_restart_path = os.path.join(checkpoints_dir, "pre_restart_v2.zip")
            if os.path.exists(pre_restart_path):
                print(f"Found pre_restart_v2.zip. Renaming to latest_model.zip to consume for resume...")
                if os.path.exists(latest_model_path):
                    os.remove(latest_model_path)
                os.rename(pre_restart_path, latest_model_path)
                
            if os.path.exists(latest_model_path):
                print(f"Resuming training from {latest_model_path} with batch_size={batch_size}")
                dml_device = torch_directml.device()
                model = PPO.load(latest_model_path, env=env, device=dml_device)
                
                # Safety Buffer: lower learning rate for the first 1000 steps of this session
                if hasattr(model.policy, "optimizer"):
                    for param_group in model.policy.optimizer.param_groups:
                        param_group['lr'] = 1e-5
                    print("Safety Buffer: Learning rate lowered to 1e-5 for the first 1000 steps.", flush=True)
                                
                # Ensure batch size is applied if reduced
                model.batch_size = batch_size 
            else:
                print(f"Starting new training session with batch_size={batch_size}")
                dml_device = torch_directml.device()
                model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="runs/v2_modular", device=dml_device, batch_size=batch_size)
            
            model_ref[0] = model
            
            simple_callback = SimpleLogCallback(
                save_freq=5000,
                save_path=checkpoints_dir,
                name_prefix="rust_rl_model",
            )

            print("Starting training... (Wait for plugin to generate data)", flush=True)
            model.learn(
                total_timesteps=1000000, 
                callback=[simple_callback, eval_callback],
                progress_bar=False,
                reset_num_timesteps=False
            )
            break # If learn completes without exception, break loop
            
        except torch.cuda.OutOfMemoryError:
            print(f"CUDA_OUT_OF_MEMORY! Attempting to drop batch_size from {batch_size} and restart.", flush=True)
            batch_size = max(4, batch_size // 2)
            torch.cuda.empty_cache()
            time.sleep(2)
            continue # Restart loop
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"CUDA_OUT_OF_MEMORY (RuntimeError)! Dropping batch_size from {batch_size} and restarting.", flush=True)
                batch_size = max(4, batch_size // 2)
                torch.cuda.empty_cache()
                time.sleep(2)
                continue
            else:
                print(f"Runtime Exception: {e}", flush=True)
                break
        except KeyboardInterrupt:
            print("Training interrupted by user.", flush=True)
            break
        except Exception as e:
            print(f"Unexpected Exception: {e}", flush=True)
            traceback.print_exc()
            break
            
    # Save the final model as 'latest_model' for future resumes
    if model_ref[0]:
        model_ref[0].save(latest_model_path)
        print(f"Model saved to {latest_model_path}", flush=True)
        
    wandb.finish()

if __name__ == "__main__":
    train()
