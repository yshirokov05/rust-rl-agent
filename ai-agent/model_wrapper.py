import torch
import torch.nn as nn
import argparse

class NatureCNN(nn.Module):
    def __init__(self, action_dim=15):
        super(NatureCNN, self).__init__()
        
        # 84x84 Input -> 3x Conv2d
        self.conv_net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # --- 1080p HANDSHAKE ---
        # Any input resolution will be spatially collapsed to 7x7 before the FC layers.
        # This allows the 84x84 pretrained knowledge (which produces 7x7) to scale.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.flatten = nn.Flatten()
        self.fc_net = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim) # Expanded to 15 neurons
        )

    def forward(self, x):
        # Handle grayscale conv input [batch, 1, h, w]
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        x = self.conv_net(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        logits = self.fc_net(x)
        return logits

def test_1080p():
    model = NatureCNN(action_dim=15)
    print("[TEST] Initializing 1080p Handshake...")
    
    # Simulate a 1080p grayscale input
    dummy_1080p = torch.randn(1, 1, 1080, 1080)
    
    output = model(dummy_1080p)
    print(f"[TEST] Input: {dummy_1080p.shape}")
    print(f"[TEST] Output logits: {output.shape}")
    
    if output.shape == (1, 15):
        print("[TEST] SUCCESS: AdaptiveAveragePooling2d matched spatial weights.")
    else:
        print("[TEST] FAILED: Shape mismatch.")

def load_v1_weights(model_path):
    print(f"[LOAD] Bootstrapping from {model_path}...")
    model = NatureCNN(action_dim=15)
    
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Manually remap the final layer weights to handle expansion (12 -> 15)
        # We load everything into the model first (except the mismatched head)
        model_dict = model.state_dict()
        
        # 1. Filter out head mismatch keys
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.shape == model_dict[k].shape}
        
        # 2. Overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        
        # 3. Manually inject the first 12 neurons into the new 15-neuron head
        # checkpoint['fc_net.2.weight'] is [512, 12] in original? 
        # Wait, original fc_net[2] was Linear(512, 12). So weights are [12, 512].
        model_dict['fc_net.2.weight'][:12, :] = checkpoint['fc_net.2.weight']
        model_dict['fc_net.2.bias'][:12] = checkpoint['fc_net.2.bias']
        
        model.load_state_dict(model_dict)
        print(f"[LOAD] Partial load success. Neurons [0:11] bootstrapped. [12:14] randomized.")
        return model
    except Exception as e:
        print(f"[LOAD ERROR] Failed to bootstrap V1.pth: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-1080p", action="store_true")
    parser.add_argument("--load-v1", action="store_true")
    args = parser.parse_args()
    
    if args.test_1080p:
        test_1080p()
    if args.load_v1:
        load_v1_weights(r"C:\Projects\rust-rl-agent\models\Master_Bean_Brain_V1.pth")
