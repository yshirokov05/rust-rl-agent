import json
import numpy as np
from PIL import Image
import os

def generate():
    path = r"C:\Projects\rust-rl-agent\shared-data\vision_0.json"
    if not os.path.exists(path):
        print(f"ERROR: {path} not found.")
        return

    with open(path, "r") as f:
        data = json.load(f)

    if "DepthMatrix" not in data:
        print("ERROR: No DepthMatrix in vision data.")
        # Print keys to debug
        print(f"Available keys: {list(data.keys())}")
        return

    matrix = np.array(data["DepthMatrix"]).reshape((84, 84))
    # Normalize and convert to grayscale
    img_data = (matrix * 255).astype(np.uint8)
    img = Image.fromarray(img_data, mode="L")
    
    out_path = "sample_vision.png"
    img.save(out_path)
    print(f"SUCCESS: Saved {out_path}")

if __name__ == "__main__":
    generate()
