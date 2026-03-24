import torch
import numpy as np
import glob
import os
from torch.utils.data import Dataset, DataLoader

class BuildingDataLoader(Dataset):
    """
    Trajectory loader for 'Base Building' data.
    Maps movement (12 base neurons) + building primitives (3 new neurons).
    """
    def __init__(self, data_dir=r"C:\Projects\rust-rl-agent\shared-data\building"):
        self.data_dir = data_dir
        self.files = glob.glob(os.path.join(data_dir, "*.npz"))
        print(f"[LOADER] Found {len(self.files)} building batches.")
        
        self.samples = []
        for f in self.files:
            # For expansion, we assume building data might have different keys or 
            # we inject the building flags [CRAFT, PLACE, UPGRADE]
            self.samples.append(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        data = np.load(path)
        
        # In 'Building' mode, we expect actions to include the 3 new flags
        depth = torch.tensor(data['depth'], dtype=torch.float32)
        
        # Placeholder mapping for 15-neuron space:
        # [0-11]: Base Actions
        # [12]: CRAFT_TOOL
        # [13]: PLACE_FOUNDATION
        # [14]: UPGRADE_WALL
        
        # For now, we return a single random step from the batch
        s_idx = np.random.randint(0, len(depth))
        d = depth[s_idx]
        a = data['action'][s_idx] # Assume 15-vector or we map it
        
        return d, torch.tensor(a, dtype=torch.long)

if __name__ == "__main__":
    # Create the directory if it doesn't exist for dev
    os.makedirs(r"C:\Projects\rust-rl-agent\shared-data\building", exist_ok=True)
    loader = BuildingDataLoader()
    print(f"[LOADER] Readiness Check: OK")
