import sys
import tempfile
import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ai-agent"))

from environment import IMAGE_SHAPE, VECTOR_SIZE
from protocol import ACTION_SIZE
from train_resnet_v2 import RustFeaturesExtractor


class TinyRustLikeEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ACTION_SIZE,), dtype=np.float32
        )
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0, high=255, shape=IMAGE_SHAPE, dtype=np.uint8
                ),
                "vector": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(VECTOR_SIZE,),
                    dtype=np.float32,
                ),
            }
        )
        self.steps = 0

    def _observation(self):
        return {
            "image": np.zeros(IMAGE_SHAPE, dtype=np.uint8),
            "vector": np.zeros(VECTOR_SIZE, dtype=np.float32),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return self._observation(), {}

    def step(self, action):
        self.steps += 1
        reward = float(np.asarray(action, dtype=np.float32)[1])
        return self._observation(), reward, False, self.steps >= 8, {}


class StableBaselinesSmokeTests(unittest.TestCase):
    def test_tiny_ppo_learn_save_load_predict(self):
        env = TinyRustLikeEnv()
        check_env(env, warn=True)
        policy_kwargs = {
            "features_extractor_class": RustFeaturesExtractor,
            "normalize_images": False,
            "net_arch": {"pi": [16], "vf": [16]},
        }
        model = PPO(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs,
            n_steps=8,
            batch_size=8,
            n_epochs=1,
            device="cpu",
            verbose=0,
        )
        model.learn(total_timesteps=16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model"
            model.save(path)
            loaded = PPO.load(path, device="cpu")
            action, _ = loaded.predict(env.reset()[0], deterministic=True)
        self.assertEqual(action.shape, (ACTION_SIZE,))
        self.assertTrue(np.isfinite(action).all())


if __name__ == "__main__":
    unittest.main()
