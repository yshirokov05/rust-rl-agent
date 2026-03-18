import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
import os
import time

class RustEnv(gym.Env):
    """
    A simplified Gymnasium environment for a Rust AI agent.
    Reads vision data from a shared JSON file.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, vision_path="../shared-data/vision.json"):
        super(RustEnv, self).__init__()
        self.vision_path = vision_path

        # Define action space: [Forward/Backward, Left/Right, Jump, Attack/Interact]
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

        # Observation space: 
        # [PlayerPos(x,y,z), NearestTree(x,y,z), NearestOre(x,y,z)]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
        )

    def _get_obs(self):
        if not os.path.exists(self.vision_path):
            return np.zeros(9, dtype=np.float32)

        try:
            with open(self.vision_path, 'r') as f:
                data = json.load(f)
            
            player = data.get('PlayerPosition') or {'X': 0, 'Y': 0, 'Z': 0}
            tree = data.get('NearestTree') or {'X': 0, 'Y': 0, 'Z': 0}
            ore = data.get('NearestOre') or {'X': 0, 'Y': 0, 'Z': 0}

            obs = np.array([
                player['X'], player['Y'], player['Z'],
                tree['X'], tree['Y'], tree['Z'],
                ore['X'], ore['Y'], ore['Z']
            ], dtype=np.float32)
            return obs
        except (json.JSONDecodeError, KeyError, PermissionError):
            # PermissionError can happen if the plugin is writing while we read
            return np.zeros(9, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
        # Action space: [Forward/Backward, Left/Right, Jump, Attack/Interact]
        # In a real scenario, this would send actions to the game.
        
        # Wait a bit to simulate game tick
        time.sleep(0.05) 

        observation = self._get_obs()
        
        # Simple reward: 1 / (distance to nearest tree or ore + 1)
        tree_dist = np.linalg.norm(observation[3:6])
        ore_dist = np.linalg.norm(observation[6:9])
        
        # We want to minimize distance, so reward is inverse of distance
        reward = 1.0 / (min(tree_dist, ore_dist) + 1.0)
        
        # Add a small penalty for each step to encourage efficiency
        reward -= 0.01

        # Done condition (placeholder)
        terminated = False
        truncated = False
        
        info = {}

        return observation, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass

if __name__ == "__main__":
    env = RustEnv()
    obs, info = env.reset()
    print(f"Initial Observation: {obs}")
    
    # Take a random action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step Observation: {obs}")
