import json
import os
import numpy as np
import matplotlib.pyplot as plt

VISION_PATH = r"C:\Projects\rust-rl-agent\shared-data\vision_0.json"
OUTPUT_IMG = r"C:\Projects\rust-rl-agent\sample_vision.png"

def generate_sample():
    if not os.path.exists(VISION_PATH):
        print(f"ERROR: No vision file at {VISION_PATH}")
        return

    with open(VISION_PATH, "r") as f:
        data = json.load(f)

    if "DepthMatrix" not in data:
        print("ERROR: No DepthMatrix in vision data.")
        return

    matrix_flat = data["DepthMatrix"]
    print(f"Matrix Size: {len(matrix_flat)}")
    
    # Reshape to 84x84
    matrix_2d = np.array(matrix_flat).reshape(84, 84)

    # Save as Grayscale .png
    plt.imsave(OUTPUT_IMG, matrix_2d, cmap='gray')
    print(f"SUCCESS: Sample vision saved to {OUTPUT_IMG}")

if __name__ == "__main__":
    generate_sample()
