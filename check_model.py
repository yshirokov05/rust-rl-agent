import os
from stable_baselines3 import PPO

latest_model_path = "models/v2_checkpoints/latest_model.zip"
if os.path.exists(latest_model_path):
    model = PPO.load(latest_model_path)
    print(f"Model num_timesteps: {model.num_timesteps}")
    print(f"Model total_timesteps: {getattr(model, 'total_timesteps', 'N/A')}")
else:
    print("Model not found.")
