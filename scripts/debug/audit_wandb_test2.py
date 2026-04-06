import wandb

api = wandb.Api(timeout=30)
try:
    r1 = api.run("yshirokov05-personal/rust-rl-v2-modular/fhqzqe7d")
    print(f"fhqzqe7d step: {r1.summary.get('_step', 'N/A')}, state: {r1.state}")
except Exception as e:
    print(e)
    
try:
    r2 = api.run("yshirokov05-personal/rust-rl-v2-modular/deft-disco-1")
    print(f"deft-disco-1 step: {r2.summary.get('_step', 'N/A')}, state: {r2.state}")
except Exception as e:
    print(e)
