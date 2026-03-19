import torch_directml
import torch

print("=== DirectML Hardware Audit ===")
print(f"torch_directml.is_available(): {torch_directml.is_available()}")
print(f"torch_directml.device_count(): {torch_directml.device_count()}")
print(f"torch_directml.device():       {torch_directml.device()}")
try:
    print(f"torch_directml.device_name(0): {torch_directml.device_name(0)}")
except Exception as e:
    print(f"torch_directml.device_name(0): ERROR - {e}")

dml = torch_directml.device()
print(f"device.type:                   {dml.type}")

# Quick tensor compute test on DML
a = torch.randn(1024, 1024, device=dml)
b = torch.randn(1024, 1024, device=dml)
c = torch.matmul(a, b)
print(f"MatMul 1024x1024 on DML:       {c.shape} (device={c.device})")
print("=== END AUDIT ===")
