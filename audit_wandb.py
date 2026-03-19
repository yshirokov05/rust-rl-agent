import wandb
import json
import statistics
import time

api = wandb.Api(timeout=30)

run_path = "yshirokov05-personal/rust-rl-v2-modular/fhqze7d"
run = api.run(run_path)

# Retrieve the latest 50 entries
history = run.history(samples=50)

# 1. Continuity
try:
    with open("c:/Projects/rust-rl-agent/project_state.json", "r") as f:
        state = json.load(f)
        last_step = state.get("last_successful_step_count", 0)
except Exception as e:
    last_step = 0

current_step = run.summary.get("_step", 0)

# 2. Behavioral Pulse
if "action_metrics/harvest_rate" in history.columns:
    harvest_rates = history["action_metrics/harvest_rate"].dropna().tail(10).tolist()
    mean_harvest_rate = statistics.mean(harvest_rates) if harvest_rates else 0.0
else:
    mean_harvest_rate = "N/A"

# 3. Hardware Health
gpu_keys = [k for k in history.columns if "gpu" in k.lower() and "util" in k.lower()]
if gpu_keys:
    col = gpu_keys[0]
    gpu_vals = history[col].dropna().tolist()
    last_gpu_util = gpu_vals[-1] if gpu_vals else 0
else:
    last_gpu_util = "N/A"

# 4. Learning Rate
if "train/learning_rate" in history.columns:
    lr_vals = history["train/learning_rate"].dropna().tolist()
    latest_lr = lr_vals[-1] if lr_vals else "N/A"
else:
    latest_lr = "N/A"

print(f"--- DATA AUDIT ---")
print(f"Last Project State Step: {last_step}")
print(f"Current W&B Step: {current_step}")
print(f"Mean Harvest Rate (last 10): {mean_harvest_rate}")
print(f"Latest GPU Util: {last_gpu_util}")
print(f"Latest Learning Rate: {latest_lr}")

print("LATEST ENTRIES")
# Iterate dynamically 
tail_data = history.tail(3).fillna("N/A")
for idx, row in tail_data.iterrows():
    step = row.get("_step", "N/A")
    # For reward
    reward = row.get("reward_step", "N/A")
    if reward == "N/A":
        reward = row.get("rollout/ep_rew_mean", "N/A")
    hr = row.get("action_metrics/harvest_rate", "N/A")
    print(f"{step}|{reward}|{hr}")
