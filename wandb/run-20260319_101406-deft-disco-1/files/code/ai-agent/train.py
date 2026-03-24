import gymnasium as gym
import time
import os
import io
import json
import socket
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

VISION_PATH = r"C:\Projects\rust-rl-agent\shared-data\vision.json"
ACTIONS_PATH = r"C:\Projects\rust-rl-agent\shared-data\actions.json"

auto_restarting = False

def safe_load_json(path):
    try:
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
    except:
        pass
    return {}

def wait_for_server(host="127.0.0.1", port=28015):
    print(f"Checking for Rust game server/plugin on {host}:{port} (TCP)...", flush=True)
    while True:
        try:
            with socket.create_connection((host, port), timeout=5):
                print(f"SUCCESS: Rust server is ONLINE on port {port}. Proceeding to train.", flush=True)
                break
        except Exception:
            print(f"CRITICAL: SERVER OFFLINE (Port {port} refused). Waiting 30 seconds before retrying...", flush=True)
            time.sleep(30)

# Global frame buffer for GIF accumulation
gif_frames = []
gif_lock = threading.Lock()

def capture_raycast_frame():
    """Captures a single raycast map frame and appends it to the GIF buffer."""
    try:
        vision_data = safe_load_json(VISION_PATH)
        action_data = safe_load_json(ACTIONS_PATH)
        if not vision_data or not action_data:
            return
            
        px = vision_data.get("PlayerPosition", {}).get("X", 0)
        pz = vision_data.get("PlayerPosition", {}).get("Z", 0)
        
        tx = vision_data.get("NearestTree", {}).get("Position", {}).get("X", 0)
        tz = vision_data.get("NearestTree", {}).get("Position", {}).get("Z", 0)
        
        ox = vision_data.get("NearestOre", {}).get("Position", {}).get("X", 0)
        oz = vision_data.get("NearestOre", {}).get("Position", {}).get("Z", 0)
        
        is_harvesting = action_data.get("Attack", False)
        
        fig, ax = plt.subplots(figsize=(4, 4), dpi=80)
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        
        dist_tree = np.linalg.norm([px - tx, pz - tz])
        dist_ore = np.linalg.norm([px - ox, pz - oz])
        nearest_x, nearest_z = (tx, tz) if dist_tree < dist_ore else (ox, oz)
        
        tree_color = 'yellow' if (is_harvesting and dist_tree < dist_ore) else 'green'
        ore_color = 'yellow' if (is_harvesting and dist_tree >= dist_ore) else 'saddlebrown'
        
        if is_harvesting:
            target = (tx, tz) if dist_tree < dist_ore else (ox, oz)
            ax.scatter(*target, color='yellow', s=500, alpha=0.5)
                    
        ax.scatter(px, pz, color='dodgerblue', s=200, label='Agent')
        ax.scatter(tx, tz, color=tree_color, s=150, marker='^', label='Tree/Hemp')
        ax.scatter(ox, oz, color=ore_color, s=150, marker='s', label='Ore')
        ax.plot([px, nearest_x], [pz, nearest_z], color='lime' if is_harvesting else 'white', linestyle='--', alpha=0.5)
        
        all_x = [px, tx, ox]
        all_z = [pz, tz, oz]
        ax.set_xlim(min(all_x) - 5.0, max(all_x) + 5.0)
        ax.set_ylim(min(all_z) - 5.0, max(all_z) + 5.0)
        
        ax.legend(facecolor='#1e1e1e', edgecolor='none', labelcolor='white', loc='upper left', fontsize=7)
        ax.grid(color='gray', linestyle=':', alpha=0.2)
        ax.set_title("Raycast & Harvest Targeting", color='white', fontsize=9)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        
        # Convert to PIL Image for GIF assembly
        from PIL import Image
        img = Image.open(buf).copy()
        with gif_lock:
            gif_frames.append(img)
            # Keep only the last 20 frames (~10 seconds of data at 2fps capture rate)
            if len(gif_frames) > 20:
                gif_frames.pop(0)
    except Exception:
        pass

def flush_gif_to_wandb():
    """Assembles accumulated frames into a GIF and uploads to W&B."""
    with gif_lock:
        if len(gif_frames) < 2:
            return
        frames_copy = list(gif_frames)
    
    try:
        gif_buf = io.BytesIO()
        frames_copy[0].save(
            gif_buf, format='GIF', save_all=True, 
            append_images=frames_copy[1:], duration=500, loop=0
        )
        gif_buf.seek(0)
        wandb.log({"AMD_GPU_Verification/raycast_gif": wandb.Video(gif_buf, format="gif", fps=2)}, commit=True)
        print(f"  >> GIF uploaded to W&B ({len(frames_copy)} frames)", flush=True)
    except Exception as e:
        print(f"  >> GIF upload failed: {e}", flush=True)

def background_frame_capture_thread():
    """Captures raycast frames every 2.0 seconds for GIF assembly (lowered to reduce CPU bottleneck)."""
    while True:
        time.sleep(2.0)
        capture_raycast_frame()

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
            
            wandb.log({
                "pulse/sps": sps,
                "custom/hardware_green": 1,   # DML confirmed active
                "custom/software_green": 1,   # PPO loop running
            }, commit=True)
            
            # Log device info for W&B system panel
            device_type = str(model.device) if hasattr(model, 'device') else 'unknown'
            gpu_load = f"DirectML ({device_type})"
                
            # Get latest reward from the SB3 logger if possible
            last_reward = 0.0
            if model.logger and model.logger.name_to_value:
                last_reward = model.logger.name_to_value.get("rollout/ep_rew_mean", 0.0)
                
            wandb.log({
                "system/device_type": device_type,
                "system/batch_size": model.batch_size if hasattr(model, 'batch_size') else 0,
                "system/n_steps": model.n_steps if hasattr(model, 'n_steps') else 0,
            }, commit=True)
            
            # SPS FALLBACK: If SPS < 10, force CPU fallback for comparison
            # [DISABLED] Auto-throttler removed to maintain user-requested hardware saturation
            # if sps > 0 and sps < 10 and last_step >= 0:
            #     print(f"WARNING: SPS={sps:.1f} < 10! Dropping batch_size to 32 for CPU fallback comparison.", flush=True)
            #     if hasattr(model, 'batch_size'):
            #         model.batch_size = 32
            #     wandb.log({"custom/hardware_green": 0, "alert/sps_fallback": 1}, commit=True)
            pass
                
            print(f"Status: Training Active. Step: {current_step}. SPS: {sps:.1f}. Last Reward: {last_reward:.1f}. Device: {gpu_load}", flush=True)
            
            if current_step == last_step:
                stall_timer += 30
                if stall_timer >= 300:
                    print(f"ERROR: Step count has not moved from {current_step} for 5 minutes. Stalled. Triggering watchdog auto-restart.", flush=True)
                    global auto_restarting
                    auto_restarting = True
                    import _thread
                    _thread.interrupt_main()
            else:
                last_step = current_step
                stall_timer = 0
        except Exception as e:
            pass

class SimpleLogCallback(CheckpointCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.harvest_history = []
        self.harvest_rate_log = []  # For console dump

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
            
        if not hasattr(self, "lr_restored") and self.num_timesteps > 2000 and hasattr(self, "model") and hasattr(self.model.policy, "optimizer"):
            for param_group in self.model.policy.optimizer.param_groups:
                if param_group['lr'] == 1e-5:
                    param_group['lr'] = 3e-4
                    print(f"\n*** SAFETY BUFFER COMPLETE: Restored learning rate to 3e-4 at total step {self.num_timesteps} ***\n", flush=True)
                    self.lr_restored = True
        
        # Every 50 steps: log metrics WITH commit=True to force cloud flush
        if self.n_calls % 50 == 0:
            harvest_rate = sum(self.harvest_history) / len(self.harvest_history) if self.harvest_history else 0.0
            hr_pct = harvest_rate * 100.0
            self.harvest_rate_log.append(hr_pct)
            if len(self.harvest_rate_log) > 20:
                self.harvest_rate_log.pop(0)
            
            log_dict = {
                "inventory/wood": wood_count,
                "inventory/cloth": cloth_count,
                "reward_step": reward,
                "action_metrics/harvest_rate": hr_pct,
                "global_step": self.num_timesteps,
            }
            
            # Pull loss metrics from SB3 logger if available
            if hasattr(self, "model") and self.model.logger and self.model.logger.name_to_value:
                logger_vals = self.model.logger.name_to_value
                pg_loss = logger_vals.get("train/policy_gradient_loss", None)
                v_loss = logger_vals.get("train/value_loss", None)
                ent = logger_vals.get("train/entropy_loss", None)
                if pg_loss is not None:
                    log_dict["train/policy_gradient_loss"] = pg_loss
                if v_loss is not None:
                    log_dict["train/value_loss"] = v_loss
                if ent is not None:
                    log_dict["train/entropy_loss"] = ent
            
            # Upload single raycast image to Live_Action_View
            if self.n_calls % 100 == 0:
                try:
                    with gif_lock:
                        if gif_frames:
                            buf = io.BytesIO()
                            gif_frames[-1].save(buf, format='PNG')
                            buf.seek(0)
                            log_dict["Live_Action_View"] = wandb.Image(buf, caption=f"Step {self.num_timesteps}")
                except Exception:
                    pass
            
            # Let W&B batch these to save CPU overhead
            wandb.log(log_dict)
        
        # Every 1000 steps: flush the GIF + print harvest_rate to console + FORCE W&B CLOUD SYNC
        if self.n_calls % 1000 == 0:
            flush_gif_to_wandb()
            last_5 = self.harvest_rate_log[-5:] if len(self.harvest_rate_log) >= 5 else self.harvest_rate_log
            print(f"  >> Last 5 harvest_rate values: {[f'{v:.1f}%' for v in last_5]}", flush=True)
            
            # FORCE SYNC: Ensure cloud dashboard updates even with large batch sizes
            print(f"  >> [W&B HEARTBEAT] Syncing Step {self.num_timesteps} to Cloud Dashboard...", flush=True)
            wandb.log({"pulse/cloud_heartbeat": 1}, commit=True)
            
            if last_5 and all(v == 0 for v in last_5):
                print("  >> WARNING: All harvest_rates are 0! Model may have collapsed. Consider increasing ent_coef.", flush=True)
        
        return super()._on_step()

def train():
    wait_for_server()
    models_dir = "models"
    checkpoints_dir = os.path.join(models_dir, "v2_checkpoints")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(os.path.dirname(VISION_PATH), exist_ok=True)
    
    run = wandb.init(
        project="rust-rl-v2-modular",
        name="Overnight_Final_Success",
        mode="online", # Synced to W&B cloud dashboard
        sync_tensorboard=False, # Disabled — we log directly to avoid step conflicts
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
    
    batch_size = 1024  # Extreme scaling for GPU saturation
    n_steps = 1024     # High-frequency updates for GPU visual feedback
    model_ref = [None] # Array ref so pulse thread can see the model
    
    # Start background frame capture thread (lightweight, no W&B logging inside)
    t_dash = threading.Thread(target=background_frame_capture_thread, daemon=True)
    t_dash.start()

    
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
                print(f"Resuming training from {latest_model_path} with batch_size={batch_size}, n_steps={n_steps}")
                dml_device = torch_directml.device()
                assert dml_device.type == 'privateuseone', f"HARD CRASH: DirectML device not found! Got device type '{dml_device.type}'. Fix your AMD driver!"
                print(f"DirectML ASSERT PASSED: device.type = '{dml_device.type}'", flush=True)
                model = PPO.load(latest_model_path, env=env, device=dml_device)
                model.n_steps = n_steps
                model.batch_size = batch_size
                model.n_epochs = 20
                
                # Safety Buffer: lower learning rate for the first 1000 steps of this session
                if hasattr(model.policy, "optimizer"):
                    for param_group in model.policy.optimizer.param_groups:
                        param_group['lr'] = 1e-5
                    print("Safety Buffer: Learning rate lowered to 1e-5 for the first 1000 steps.", flush=True)
            else:
                print(f"Starting new training session with batch_size={batch_size}, n_steps={n_steps}")
                dml_device = torch_directml.device()
                assert dml_device.type == 'privateuseone', f"HARD CRASH: DirectML device not found! Got device type '{dml_device.type}'. Fix your AMD driver!"
                print(f"DirectML ASSERT PASSED: device.type = '{dml_device.type}'", flush=True)
                model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="runs/v2_modular", device=dml_device, batch_size=batch_size, n_steps=n_steps, n_epochs=20, ent_coef=0.05)
            
            model_ref[0] = model
            
            simple_callback = SimpleLogCallback(
                save_freq=25000,
                save_path=checkpoints_dir,
                name_prefix="overnight_win",
            )

            print(f"Starting training... (Current total steps: {model.num_timesteps})", flush=True)
            model.learn(
                total_timesteps=10000000, 
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
            global auto_restarting
            if auto_restarting:
                print("Watchdog triggered restart. Continuing loop...", flush=True)
                auto_restarting = False
                continue
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
