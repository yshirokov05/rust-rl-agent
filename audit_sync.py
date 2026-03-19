import wandb

api = wandb.Api()
run = api.run("yshirokov05-personal/rust-rl-v2-modular/deft-disco-1")

print(f"Run state: {run.state}")
print(f"W&B _step: {run.summary.get('_step', 'N/A')}")
print(f"global_step: {run.summary.get('global_step', 'N/A')}")
print(f"harvest_rate: {run.summary.get('action_metrics/harvest_rate', 'N/A')}")
print(f"pg_loss: {run.summary.get('train/policy_gradient_loss', 'N/A')}")
print(f"v_loss: {run.summary.get('train/value_loss', 'N/A')}")
print(f"ent_loss: {run.summary.get('train/entropy_loss', 'N/A')}")
print(f"hw_green: {run.summary.get('custom/hardware_green', 'N/A')}")
print(f"sw_green: {run.summary.get('custom/software_green', 'N/A')}")
print(f"sps: {run.summary.get('pulse/sps', 'N/A')}")

# Check history count
history = run.history(samples=10)
print(f"History rows returned: {len(history)}")
print(f"History columns: {list(history.columns)[:15]}")
