import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
import os
import time


class RustEnv(gym.Env):
    """
    A Gymnasium environment for a Rust AI agent.
    Communicates with the game server via shared JSON files:
      - Reads:  shared-data/vision.json  (bot position, nearest tree/ore, health)
      - Writes: shared-data/actions.json (forward, strafe, jump, attack)
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, vision_path="shared-data/vision.json", actions_path="shared-data/actions.json"):
        super(RustEnv, self).__init__()
        self.vision_path = vision_path
        self.actions_path = actions_path
        self.steps = 0
        self.max_steps = 2048  # Episode length
        self.prev_tree_dist = None
        self.prev_ore_dist = None

        # Ensure shared-data directory exists
        os.makedirs(os.path.dirname(self.vision_path), exist_ok=True)

        # Action space: [Forward/Backward, Left/Right, Jump, Attack/Interact]
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

        # Observation space:
        # [PlayerPos(x,y,z), NearestTree(x,y,z), NearestOre(x,y,z), Health]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

    def _send_actions(self, action):
        """Write agent actions to actions.json for the BotController plugin."""
        actions_data = {
            "Forward": float(action[0]),
            "Strafe": float(action[1]),
            "Jump": float(action[2]),
            "Attack": float(action[3])
        }
        try:
            with open(self.actions_path, 'w') as f:
                json.dump(actions_data, f)
        except PermissionError:
            pass  # Plugin is reading, try next tick

    def _get_obs(self):
        """Read vision data from vision.json written by the BotController plugin."""
        if not os.path.exists(self.vision_path):
            return np.zeros(10, dtype=np.float32)

        try:
            with open(self.vision_path, 'r') as f:
                data = json.load(f)

            player = data.get('PlayerPosition') or {'X': 0, 'Y': 0, 'Z': 0}
            tree = data.get('NearestTree') or {'X': 0, 'Y': 0, 'Z': 0}
            ore = data.get('NearestOre') or {'X': 0, 'Y': 0, 'Z': 0}
            health = data.get('Health', 100.0)

            obs = np.array([
                player['X'], player['Y'], player['Z'],
                tree['X'], tree['Y'], tree['Z'],
                ore['X'], ore['Y'], ore['Z'],
                health / 100.0  # Normalize health to 0-1
            ], dtype=np.float32)
            return obs
        except (json.JSONDecodeError, KeyError, PermissionError, ValueError):
            return np.zeros(10, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.prev_tree_dist = None
        self.prev_ore_dist = None

        # Send a "stop" action to reset the bot's momentum
        self._send_actions(np.zeros(4, dtype=np.float32))
        time.sleep(0.15)  # Wait for server tick

        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
        self.steps += 1

        # Send actions to the game server
        self._send_actions(action)

        # Wait for the server to process the action (sync with 10 ticks/sec)
        time.sleep(0.1)

        # Read the new observation
        observation = self._get_obs()

        # Calculate reward
        tree_dist = np.linalg.norm(observation[3:6])
        ore_dist = np.linalg.norm(observation[6:9])
        min_dist = min(tree_dist, ore_dist)

        reward = 0.0

        # Reward for getting closer to resources
        if self.prev_tree_dist is not None:
            prev_min = min(self.prev_tree_dist, self.prev_ore_dist)
            # Positive reward for moving closer, negative for moving away
            reward += (prev_min - min_dist) * 0.5

        # Bonus for being very close to a resource (within gather range)
        if min_dist < 3.0:
            reward += 1.0

        # Bonus for attacking when near a resource
        if min_dist < 3.0 and action[3] > 0.5:
            reward += 2.0

        # Small step penalty to encourage efficiency
        reward -= 0.01

        # Update previous distances
        self.prev_tree_dist = tree_dist
        self.prev_ore_dist = ore_dist

        # Episode ends after max_steps
        truncated = self.steps >= self.max_steps
        terminated = False

        info = {
            "tree_dist": tree_dist,
            "ore_dist": ore_dist,
            "steps": self.steps
        }

        return observation, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        # Send stop action on close
        self._send_actions(np.zeros(4, dtype=np.float32))


if __name__ == "__main__":
    env = RustEnv()
    obs, info = env.reset()
    print(f"Initial Observation: {obs}")

    # Take a random action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step Observation: {obs}")
    print(f"Reward: {reward}")
    print(f"Info: {info}")
