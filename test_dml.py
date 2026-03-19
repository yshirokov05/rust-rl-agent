import torch
import torch_directml
device = torch_directml.device()
print(f"Device: {device}")
x = torch.ones(512, 512).to(device)
y = torch.ones(512, 512).to(device)
z = torch.matmul(x, y)
print(f"Matmul on {device} success. Z shape: {z.shape}")
print(f"Z device: {z.device}")
