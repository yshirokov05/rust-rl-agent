import wandb

api = wandb.Api(timeout=30)
runs = api.runs("yshirokov05-personal/rust-rl-v2-modular")
for r in runs:
    print(f"Name: {r.name}, ID: {r.id}, Path: {r.path}")
