import os
import multiprocessing
import torch
import glob
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)

try:
    import torch_directml
    _has_dml = True
except ImportError:
    _has_dml = False

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
import gymnasium as gym
from gymnasium import spaces
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np

from environment import RustEnv
from reward_shaping import RewardShaper

class ShapedRustEnv(RustEnv):
    def __init__(self, bot_id=0):
        super().__init__(bot_id=bot_id)
        self.shaper = RewardShaper()

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        
        # Inject Phase 3 Reward Shaping
        shaping_reward = self.shaper.get_shaping_reward(observation, info)
        
        return observation, reward + shaping_reward, terminated, truncated, info

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
        env = ShapedRustEnv(bot_id=bot_id)
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

    num_envs = 6 # OPTIMIZED: Matches i5-8600k physical core count (6C/6T)

    # 2. Vectorized Multi-Processing Environment
    env_fns = [make_env(i) for i in range(num_envs)]
    
    # Use SubprocVecEnv for parallel execution
    env = SubprocVecEnv(env_fns, start_method='spawn')
    env = VecMonitor(env)

    # 3. Memory-Safe Hyperparameters for 8GB VRAM
    n_steps = 512              
    batch_size = 256           # OPTIMIZED: Higher throughput for RX 5700 XT (8GB VRAM)
    buffer_size = n_steps * num_envs 

    policy_kwargs = dict(
        features_extractor_class=ResNet18FeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=[256, 128], 
    )

    torch.set_default_dtype(torch.float32) 
    
    # --- DYNAMIC BRAIN LOADING (PHASE 3) ---
    checkpoint_dir = os.path.join(_PROJECT_ROOT, "models", "checkpoints_v2")
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "vanguard_v2_*.zip"))
    
    if checkpoints:
        # Extract step counts to find the true latest save
        latest_checkpoint = max(checkpoints, key=lambda x: int(os.path.basename(x).split('_')[-2]))
        print(f"--- [LOADING BRAIN] --- Found overnight progress: {latest_checkpoint}")
        model = PPO.load(
            latest_checkpoint,
            env=env,
            device=device,
            tensorboard_log=os.path.join(_PROJECT_ROOT, "runs"),
            ent_coef=0.01
        )
    else:
        print("--- [INITIALIZING NEW BRAIN] --- No V2 checkpoints found. Starting from scratch.")
        model = PPO(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs,
            n_steps=n_steps,
            batch_size=batch_size,
            learning_rate=3e-4,
            ent_coef=0.01,
            device=device,
            verbose=1,
            tensorboard_log=os.path.join(_PROJECT_ROOT, "runs")
        )

    print(f"Starting Training V2 | Workers: {num_envs} | Model: ResNet18 RGB | BS: {batch_size} | VRAM Lock: ON")
    
    # 4. Phase 3: Checkpoint and Tensorboard Callbacks
    custom_callback = TensorboardCallback()
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=os.path.join(_PROJECT_ROOT, "models", "checkpoints_v2"),
        name_prefix="vanguard_v2"
    )
    
    callback = CallbackList([custom_callback, checkpoint_callback])
    
    try:
        model.learn(total_timesteps=10_000_000, callback=callback)
    except KeyboardInterrupt:
        print("\n--- [TERMINATION DETECTED] --- Graceful shutdown initiated...")
    finally:
        save_path = os.path.join(_PROJECT_ROOT, "models", "rust_vanguard_resnet_v2_emergency_save")
        model.save(save_path)
        print(f"--- [SAFETY FIRST] --- Brain state secured to: {save_path}")
        env.close()
