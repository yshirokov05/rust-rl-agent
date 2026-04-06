import wandb

api = wandb.Api()
run = api.run("yshirokov05-personal/rust-rl-v2-modular/deft-disco-1")

history = run.history(samples=50)

if "pulse/sps" in history.columns:
    sps_list = history["pulse/sps"].dropna().tail(5).tolist()
    print("Latest SPS values:", sps_list)
    if sps_list and sps_list[-1] > 50:
        print("PASS! SPS is above 50!")
    else:
        print("FAIL! SPS is below 50.")
else:
    print("No SPS found yet.")
