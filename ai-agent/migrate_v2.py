import torch
from stable_baselines3 import PPO
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))
from environment import RustEnv
from stable_baselines3.common.monitor import Monitor
from gymnasium import spaces
import numpy as np
import os
import shutil

def migrate_to_v2():
    # old model had 6 actions
    old_path = "models/latest_model.zip"
    
    # new model has 7 actions, user wants it in v2_checkpoints
    v2_dir = "models/v2_checkpoints"
    os.makedirs(v2_dir, exist_ok=True)
    new_path = os.path.join(v2_dir, "latest_model.zip")
    
    if not os.path.exists(old_path):
        print(f"Old model not found at {old_path}")
        # fallback, just initialize a fresh 7 action model and save it there so the dashboard works
        env = RustEnv()
        model = PPO("MlpPolicy", env, verbose=0)
        model.save(new_path)
        print("Created fresh 7-action baseline model.")
        return

    print("--- Performing Neural Surgery (6 -> 7 Actions) ---")
    
    # Mock the old 6 action env
    from gymnasium.envs.registration import EnvSpec
    class MockEnv(RustEnv):
        def __init__(self):
            super().__init__()
            self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
            
    env_old = MockEnv()
    old_model = PPO.load(old_path, env=env_old, device="cpu")
    old_params = old_model.policy.state_dict()
    
    # Create the new env with 7 actions
    env_new = RustEnv()
    new_model = PPO("MlpPolicy", env_new, verbose=0, device="cpu")
    new_params = new_model.policy.state_dict()
    
    # Migrate weights
    for key in new_params:
        if key in old_params:
            if new_params[key].shape == old_params[key].shape:
                new_params[key] = old_params[key]
            else:
                if "action_net" in key:
                    old_layer = old_params[key]
                    new_layer = new_params[key]
                    if len(old_layer.shape) == 2: # Weight
                        new_layer[:old_layer.shape[0], :] = old_layer
                    else: # Bias
                        new_layer[:old_layer.shape[0]] = old_layer
                elif "log_std" in key:
                    old_std = old_params[key]
                    new_std = new_params[key]
                    new_std[:old_std.shape[0]] = old_std

    new_model.policy.load_state_dict(new_params)
    new_model.save(new_path)
    print(f"--- Surgery Complete. Model migrated to 7-action space and saved to {new_path} ---")

if __name__ == "__main__":
    migrate_to_v2()
