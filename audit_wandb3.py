import wandb
import json
import statistics

api = wandb.Api()

# Because there are two deft-disco-1 runs, we fetch from the one we created or resumed (`id="deft-disco-1"`)
# Or we just fetch from both and use the one that has higher steps
run = api.run("yshirokov05-personal/rust-rl-v2-modular/deft-disco-1")

history = run.history(samples=50)

# 1. Continuity
try:
    with open("c:/Projects/rust-rl-agent/project_state.json", "r") as f:
        state = json.load(f)
        last_step = state.get("last_successful_step_count", 0)
except Exception:
    last_step = 0

current_step = run.summary.get("_step", 0)

# 2. Behavioral Pulse
hr_col = "action_metrics/harvest_rate"
if hr_col in history.columns:
    harvest_rates = history[hr_col].dropna().tail(10).tolist()
    mean_harvest_rate = statistics.mean(harvest_rates) if harvest_rates else 0.0
else:
    mean_harvest_rate = 0.0

# 3. Hardware Health
gpu_keys = [k for k in history.columns if "gpu" in k.lower() and "util" in k.lower()]
last_gpu_util = "N/A"
if gpu_keys:
    col = gpu_keys[0]
    gpu_vals = history[col].dropna().tolist()
    if gpu_vals:
        last_gpu_util = gpu_vals[-1]
else:
    # Stable Baselines / WandB might log system metrics async 
    try:
        sys_metrics = run.history(stream="events", samples=10)
        sys_gpu_keys = [k for k in sys_metrics.columns if "gpu" in k.lower() and "util" in k.lower()]
        if sys_gpu_keys:
            vals = sys_metrics[sys_gpu_keys[0]].dropna().tolist()
            if vals: last_gpu_util = vals[-1]
    except Exception:
        pass

# 4. Learning Rate
lr_keys = [k for k in history.columns if "learning_rate" in k or "lr" in k.lower().split("/")]
latest_lr = "N/A"
if "train/learning_rate" in history.columns:
    lr_vals = history["train/learning_rate"].dropna().tolist()
    if lr_vals: latest_lr = lr_vals[-1]

print(f"LAST_PROJECT_STATE_STEP={last_step}")
print(f"CURRENT_STEP={current_step}")
print(f"MEAN_HR={mean_harvest_rate}")
print(f"LATEST_GPU_UTIL={last_gpu_util}")
print(f"LATEST_LR={latest_lr}")

print("--- TABLE DATA ---")
tail_data = history.tail(3).fillna("N/A")
for idx, row in tail_data.iterrows():
    step = row.get("_step", "N/A")
    reward = row.get("reward_step", "N/A")
    if reward == "N/A": reward = row.get("rollout/ep_rew_mean", "N/A")
    hr = row.get(hr_col, "N/A")
    print(f"{step}|{reward}|{hr}")
