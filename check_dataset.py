import numpy as np
import os
import glob

LOG_DIR = r"C:\Projects\rust-rl-agent\shared-data\depth_logs"

def check():
    files = glob.glob(os.path.join(LOG_DIR, "*.npz"))
    if not files:
        print("No files found.")
        return
    
    file = files[0]
    data = np.load(file)
    print(f"Keys in {os.path.basename(file)}: {data.files}")
    for key in data.files:
        print(f"Shape of {key}: {data[key].shape}")

if __name__ == "__main__":
    check()
