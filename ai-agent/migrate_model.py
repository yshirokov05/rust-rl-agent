import torch
from stable_baselines3 import PPO
from environment import RustEnv
from stable_baselines3.common.monitor import Monitor
from gymnasium import spaces
import numpy as np
import os

def migrate_model():
    old_path = "models/old_model_5_actions.zip"
    new_path = "models/latest_model.zip"
    
    if not os.path.exists(old_path):
        print("Old model not found.")
        return

    print("--- Performing Neural Surgery ---")
    
    env_5.action_space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)
    env_5 = Monitor(env_5)
    
    old_model = PPO.load(old_path, env=env_5)
    old_params = old_model.policy.state_dict()
    
    # 2. Create the new env with 6 actions
    env_6 = RustEnv() # Default now 6
    env_6 = Monitor(env_6)
    
    new_model = PPO("MlpPolicy", env_6, verbose=1)
    new_params = new_model.policy.state_dict()
    
    # 3. Migrate weights
    # Most weights (layers) will match exactly. 
    # Only the final output layer (action_net) will differ.
    
    for key in new_params:
        if key in old_params:
            if new_params[key].shape == old_params[key].shape:
                new_params[key] = old_params[key]
                print(f"Migrated layer: {key}")
            else:
                # This is the action_net. We copy the first 4 actions and init the 5th (Sprint).
                if "action_net" in key:
                    print(f"Splicing layer: {key}")
                    old_layer = old_params[key]
                    new_layer = new_params[key]
                    # Shape is typically [out_features, in_features]
                    # For a 4->5 action space, out_features changes.
                    # In PPO, action_net is the mean. log_std is separate.
                    if len(old_layer.shape) == 2: # Weight
                        new_layer[:old_layer.shape[0], :] = old_layer
                    else: # Bias
                        new_layer[:old_layer.shape[0]] = old_layer
                elif "log_std" in key:
                    print(f"Splicing log_std: {key}")
                    old_std = old_params[key]
                    new_std = new_params[key]
                    new_std[:old_std.shape[0]] = old_std

    # 4. Load the spliced weights
    new_model.policy.load_state_dict(new_params)
    
    # 5. Save as latest_model
    new_model.save(new_path)
    print("--- Surgery Complete. Model migrated to 5-action space. ---")

if __name__ == "__main__":
    migrate_model()
