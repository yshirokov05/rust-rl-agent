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
            
            player = data.get('PlayerPosition', {'X': 0, 'Y': 0, 'Z': 0})
            tree = data.get('NearestTree', {'X': 0, 'Y': 0, 'Z': 0})
            ore = data.get('NearestOre', {'X': 0, 'Y': 0, 'Z': 0})

            obs = np.array([
                player['X'], player['Y'], player['Z'],
                tree['X'], tree['Y'], tree['Z'],
                ore['X'], ore['Y'], ore['Z']
            ], dtype=np.float32)
            return obs
        except (json.JSONDecodeError, KeyError):
            return np.zeros(9, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
        # In a real scenario, this would send actions to the game via a socket or similar.
        # Here we just read the updated vision data.
        
        # Wait a bit to simulate game tick or wait for plugin update
        # time.sleep(0.1) 

        observation = self._get_obs()
        
        # Simple reward: 1 if we are close to a tree or ore
        # In practice, this would be much more complex.
        reward = 0.0
        
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
