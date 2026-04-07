import os
import multiprocessing
import torch

try:
    import torch_directml
    _has_dml = True
except ImportError:
    _has_dml = False

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
from gymnasium import spaces
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np

from environment import RustEnv

class ResNet18FeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        weights = ResNet18_Weights.IMAGENET1K_V1
        self.resnet = resnet18(weights=weights)
        for name, param in self.resnet.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False
        self.resnet.fc = nn.Identity()
        self._features_dim = 512 + 14

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def forward(self, observations: dict) -> torch.Tensor:
        img = observations["image"].float() / 255.0
        res_features = self.resnet(img)
        vec_features = observations["vector"]
        return torch.cat((res_features, vec_features), dim=1)

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        wood_counts = []
        has_plans = []
        for info in infos:
            if "wood_count" in info:
                wood_counts.append(info["wood_count"])
            if "has_plan" in info:
                has_plans.append(int(info["has_plan"]))
        
        if wood_counts:
            self.logger.record("custom/wood_count_avg", np.mean(wood_counts))
        if has_plans:
            self.logger.record("custom/has_plan_ratio", np.mean(has_plans))
        return True

def make_env(bot_id):
    def _init():
        env = RustEnv(bot_id=bot_id)
        return env
    return _init

if __name__ == '__main__':
    # Mandatory for Windows multiprocessing stability
    multiprocessing.freeze_support()
    
    # 1. Hardware Max-Out Config
    try:
        if _has_dml:
            device = torch_directml.device()
            print(f"Using DirectML Device: {device} (AMD RX 5700 XT)")
        else:
            raise ImportError("torch_directml not explicitly imported")
    except Exception as e:
        print(f"DirectML init failed, falling back to CPU: {e}")
        device = torch.device('cpu')

    num_envs = 6 # Production Training Mode

    # 2. Vectorized Multi-Processing Environment
    env_fns = [make_env(i) for i in range(num_envs)]
    
    # Use SubprocVecEnv to force execution in main thread for diagnostics
    env = SubprocVecEnv(env_fns, start_method='spawn')
    env = VecMonitor(env)

    # 3. Memory-Safe Hyperparameters for 8GB VRAM
    n_steps = 512              
    batch_size = 64            
    buffer_size = n_steps * num_envs 

    policy_kwargs = dict(
        features_extractor_class=ResNet18FeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=[256, 128], 
    )

    torch.set_default_dtype(torch.float32) 
    
    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        n_steps=n_steps,
        batch_size=batch_size,
        learning_rate=3e-4,
        device=device, 
        verbose=1,
        tensorboard_log="C:/Projects/ml_logs/tensorboard_logs"
    )

    print(f"Starting Training | Workers: {num_envs} | Model: ResNet18 RGB | BS: {batch_size} | VRAM Lock: ON")
    custom_callback = TensorboardCallback()
    model.learn(total_timesteps=500000, callback=custom_callback)
    model.save("C:/Projects/ml_logs/models/rust_vanguard_resnet_v1")
