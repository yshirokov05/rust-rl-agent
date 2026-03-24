import glob
import os

DATA_DIR = r"C:\Projects\rust-rl-agent\shared-data\depth_logs"
files = glob.glob(os.path.join(DATA_DIR, "**/*.npz"), recursive=True)
print(f"REAL HARD COUNT: {len(files)} .npz files total across all shards.")

# Calculate total steps (assuming 200 per batch)
total_steps = len(files) * 200
print(f"TOTAL SYNCED STEPS: {total_steps}")
