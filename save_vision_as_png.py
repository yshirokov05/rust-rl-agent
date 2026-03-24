import json
import numpy as np
from PIL import Image
import os

VISION_PATH = r"C:\Projects\rust-rl-agent\shared-data\vision_0.json"
OUTPUT_PATH = r"C:\Projects\rust-rl-agent\LIVE_VIEW.png"

def main():
    if not os.path.exists(VISION_PATH):
        print(f"Error: {VISION_PATH} not found.")
        return

    with open(VISION_PATH, 'r') as f:
        data = json.load(f)
    
    depth_matrix = data.get('DepthMatrix', [])
    if not depth_matrix:
        print("Error: No DepthMatrix found in JSON.")
        return
        
    # Assuming 84x84 = 7056
    matrix = np.array(depth_matrix).reshape((84, 84))
    
    # Normalize 0-1 to 0-255
    rescaled = (matrix * 255).astype(np.uint8)
    
    img = Image.fromarray(rescaled, 'L')
    img = img.resize((512, 512), Image.NEAREST) # Upscale for visibility
    img.save(OUTPUT_PATH)
    print(f"Success: Saved {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
