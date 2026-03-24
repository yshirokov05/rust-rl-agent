import json
import numpy as np
import matplotlib.pyplot as plt
import os

VISION_PATH = r"C:\Projects\rust-rl-agent\shared-data\vision_0.json"
OUTPUT_PATH = r"C:\Projects\rust-rl-agent\visual_audit\bot_0_vision_live.png"

def render_vision():
    if not os.path.exists(VISION_PATH):
        print("Vision file not found.")
        return

    with open(VISION_PATH, 'r') as f:
        data = json.load(f)
    
    if "DepthMatrix" not in data:
        print("DepthMatrix not found in JSON.")
        return

    matrix = np.array(data["DepthMatrix"]).reshape(84, 84)
    
    plt.figure(figsize=(6,6))
    plt.imshow(matrix, cmap='viridis')
    plt.colorbar(label='Depth')
    plt.title("Bot 0 Live Vision (Depth)")
    plt.axis('off')
    plt.savefig(OUTPUT_PATH)
    print(f"Vision rendered to {OUTPUT_PATH}")

if __name__ == "__main__":
    render_vision()
