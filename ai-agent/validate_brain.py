import torch
import numpy as np
import glob
import os
import random
import torch.nn as nn
import torch_directml

# --- ARCHITECTURE (Duplicate from bc_nature_cnn.py for standalone) ---
class NatureCNN(nn.Module):
    def __init__(self, action_dim=12):
        super(NatureCNN, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_net = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )
    def forward(self, depth_matrix):
        x = depth_matrix.unsqueeze(1)
        conv_out = self.conv_net(x)
        logits = self.fc_net(conv_out)
        return logits

def multi_discrete_to_12(action_array):
    fwd = action_array[0] > 0
    attack = action_array[6] > 0
    jump = action_array[5] > 0
    idx = 0
    if fwd: idx += 1
    if attack: idx += 2
    if jump: idx += 4
    return min(idx, 11)

# --- VALIDATION LOGIC ---
DATA_DIR = r"C:\Projects\rust-rl-agent\shared-data"
MODEL_PATH = r"C:\Projects\rust-rl-agent\models\Master_Bean_Brain_V1.pth"
NUM_SAMPLES = 100

def validate():
    dml = torch_directml.device()
    model = NatureCNN(action_dim=12).to(dml)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    all_files = glob.glob(os.path.join(DATA_DIR, "**", "*.npz"), recursive=True)
    random.shuffle(all_files)

    correct = 0
    total = 0

    print(f"Starting Validation on {NUM_SAMPLES} samples...")
    
    with torch.no_grad():
        for path in all_files:
            if total >= NUM_SAMPLES: break
            try:
                data = np.load(path)
                depths = data['depth']
                actions = data['action']
                
                # Pick one random sample from this batch
                idx = random.randint(0, len(depths)-1)
                depth = torch.tensor(depths[idx], dtype=torch.float32).unsqueeze(0).to(dml)
                label = multi_discrete_to_12(actions[idx])
                
                logits = model(depth)
                pred = torch.argmax(logits, dim=1).item()
                
                if pred == label:
                    correct += 1
                total += 1
            except:
                continue

    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"Validation Complete.")
    print(f"Total Samples: {total}")
    print(f"Correct Predictions: {correct}")
    print(f"Top-1 Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    validate()
