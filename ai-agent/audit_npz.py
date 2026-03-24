import numpy as np
import glob
import os

DATA_DIR = r"C:\Projects\rust-rl-agent\shared-data"
files = glob.glob(os.path.join(DATA_DIR, "**/*.npz"), recursive=True)

if not files:
    print("❌ No .npz files found!")
    exit(1)

f = files[-1]
print(f"📄 Auditing File: {f}")

try:
    with np.load(f) as data:
        print(f"🔑 Keys found: {list(data.keys())}")
        if 'depth' in data:
            depth = data['depth']
            print(f"📐 Depth Matrix Shape: {depth.shape}")
            print(f"📊 Depth Max Value: {np.max(depth)}")
            print(f"📊 Depth Min Value: {np.min(depth)}")
            if np.max(depth) == 0:
                print("⚠️ WARNING: DepthMatrix is ALL ZEROS!")
            else:
                print("✅ DepthMatrix contains non-zero data.")
        else:
            print("❌ 'depth' key MISSING in .npz!")
            
        if 'pos' in data:
            print(f"📍 Position Data: {data['pos']}")
        else:
            print("❌ 'pos' key MISSING in .npz!")
except Exception as e:
    print(f"❌ Failed to load file: {e}")
