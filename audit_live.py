import wandb
import os
from datetime import datetime, timezone

def audit():
    entity = "yshirokov05-personal"
    project = "rust-rl-v2-modular"
    
    api = wandb.Api()
    print(f"Checking runs in {entity}/{project}...")
    
    try:
        runs = api.runs(f"{entity}/{project}", order="-created_at")
        if not runs:
            print("No runs found.")
            return

        print(f"\n| Run Name | ID | State | Heartbeat |")
        print(f"| --- | --- | --- | --- |")
        for run in runs[:5]:
            print(f"| {run.name} | {run.id} | {run.state} | {getattr(run, 'heartbeat_at', 'N/A')} |")
            
        # Specific check for Overnight_Final_Success
        target_run = None
        for run in runs:
            if run.name == "Overnight_Final_Success":
                target_run = run
                break
        
        if target_run:
            print(f"\nTarget Run Found: {target_run.name}")
            summary = target_run.summary
            print(f"Step: {summary.get('global_step', 'N/A')}")
            print(f"SPS: {summary.get('pulse/sps', 'N/A')}")
        else:
            print("\nCRITICAL: 'Overnight_Final_Success' run NOT FOUND yet.")
            print("Note: The training script is still waiting for the Rust server to come online.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    audit()
