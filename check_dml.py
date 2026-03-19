import torch
import traceback
print("Torch Version: ", torch.__version__)
try:
    import torch_directml
    print("DirectML Available: ", torch_directml.is_available())
    print("DirectML Device Count: ", torch_directml.device_count())
    print("DirectML Device: ", torch_directml.device())
except ImportError:
    print("DirectML not installed!")
except Exception as e:
    print(f"Exception checking DirectML: {e}")
    traceback.print_exc()
