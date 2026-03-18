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
        self.wood_count = 0
        self.cloth_count = 0
        self.tool_count = 0

        # Ensure shared-data directory exists
        os.makedirs(os.path.dirname(self.vision_path), exist_ok=True)

        # Action space: [Forward/Backward, Left/Right, Jump, Attack/Interact, Sprint]
        self.action_space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)

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
            "Attack": float(action[3]),
            "Sprint": float(action[4])
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
        self.wood_count = 0
        self.cloth_count = 0
        self.tool_count = 0

        # Send a "stop" action to reset the bot's momentum
        self._send_actions(np.zeros(5, dtype=np.float32))
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

        # Extract info from vision.json for rewards and achievements
        has_gathered = False
        try:
            if os.path.exists(self.vision_path):
                with open(self.vision_path, 'r') as f:
                    data = json.load(f)
                    has_gathered = data.get('HasGathered', False)
        except:
            pass

        # Calculate reward
        tree_dist = np.linalg.norm(observation[3:6])
        ore_dist = np.linalg.norm(observation[6:9])
        min_dist = min(tree_dist, ore_dist)

        reward = 0.0

        # Reward for getting closer to resources (+10 as requested)
        if self.prev_tree_dist is not None:
            prev_min = min(self.prev_tree_dist, self.prev_ore_dist)
            if min_dist < prev_min:
                reward += 10.0  # Constant +10 for decreasing distance
            elif min_dist > prev_min:
                reward -= 1.0   # Reduced penalty for moving away while sprinting
        
        # Reward for being very close (encourages parking/staying)
        if min_dist < 3.0:
            reward += 0.5

        # Major bonus for successful "Hit" (+100 as requested)
        if has_gathered:
            reward += 100.0

        # Small step penalty to encourage efficiency
        reward -= 0.1

        # Update previous distances
        self.prev_tree_dist = tree_dist
        self.prev_ore_dist = ore_dist

        # Episode ends after max_steps
        truncated = self.steps >= self.max_steps
        terminated = False

        # Achievements Tracking
        if has_gathered:
            if tree_dist < 3.0:
                self.wood_count += 1
            elif ore_dist < 3.0:
                # For now, let's say ore gathering counts as progress towards 'First Tool' 
                # or just track it separately. The user asked for 'First Tool'.
                # In Rust, you usually craft tools. We'll mock 'First Tool' as 
                # gathering 5 ore for now, or just leave it at 0 if not Implementable.
                # Let's just track wood and cloth for now as per heuristics.
                pass 
            else:
                # If gathered but not near tree/ore, it's likely hemp/cloth
                self.cloth_count += 1
        
        info = {
            "tree_dist": tree_dist,
            "ore_dist": ore_dist,
            "steps": self.steps,
            "has_gathered": has_gathered,
            "reward": reward,
            "wood_count": self.wood_count,
            "cloth_count": self.cloth_count,
            # Achievement logging (boolean flags for WandB)
            "achievement/10x_cloth": 1 if self.cloth_count >= 10 else 0,
            "achievement/first_wood": 1 if self.wood_count >= 1 else 0,
            "achievement/first_tool": 1 if self.tool_count >= 1 else 0
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
