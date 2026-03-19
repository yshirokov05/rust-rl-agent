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

    def __init__(self, bot_id=0, vision_path=None, actions_path=None):
        super(RustEnv, self).__init__()
        root_dir = r"C:\Projects\rust-rl-agent"
        self.vision_path = vision_path or os.path.join(root_dir, "shared-data", f"vision_{bot_id}.json")
        self.actions_path = actions_path or os.path.join(root_dir, "shared-data", f"actions_{bot_id}.json")
        self.steps = 0
        self.max_steps = 2048  # Episode length
        self.prev_tree_dist = None
        self.prev_ore_dist = None
        self.wood_count = 0
        self.cloth_count = 0
        self.tool_count = 0
        self.current_tree_name = ""
        self.current_ore_name = ""
        self.has_gathered = False

        # Ensure shared-data directory exists
        os.makedirs(os.path.dirname(self.vision_path), exist_ok=True)

        # Action space: [Forward/Backward, Left/Right, Jump, Attack/Interact, Sprint]
        # +1 for Attack/Interact = 7
        self.action_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)

        # Observation space:
        # [PlayerPos(x,y,z), NearestTree(x,y,z), NearestOre(x,y,z), Health]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

    def _send_actions(self, action):
        """Write agent actions to actions.json for the BotController plugin."""
        # Map 7 Actions
        actions = {
            "MoveX": float(action[0]),
            "MoveZ": float(action[1]),
            "LookX": float(action[2]),
            "LookY": float(action[3]),
            "Sprint": bool(action[4] > 0),
            "Jump": bool(action[5] > 0),
            "Attack": bool(action[6] > 0)
        }
        try:
            with open(self.actions_path, 'w') as f:
                json.dump(actions, f)
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
            
            tree_data = data.get('NearestTree') or {}
            tree_pos = tree_data.get('Position', {'X': 0, 'Y': 0, 'Z': 0})
            self.current_tree_name = tree_data.get('Name', '')

            ore_data = data.get('NearestOre') or {}
            ore_pos = ore_data.get('Position', {'X': 0, 'Y': 0, 'Z': 0})
            self.current_ore_name = ore_data.get('Name', '')

            health = data.get('Health', 100.0)
            self.has_gathered = data.get('HasGathered', False)

            obs = np.array([
                player['X'], player['Y'], player['Z'],
                tree_pos['X'], tree_pos['Y'], tree_pos['Z'],
                ore_pos['X'], ore_pos['Y'], ore_pos['Z'],
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
        self._send_actions(np.zeros(7, dtype=np.float32))
        # No sleep — let the training loop run at full GPU speed

        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
        self.steps += 1

        # Send actions to the game server
        self._send_actions(action)

        # No sleep — removed to allow GPU-saturated throughput (>150 SPS target)

        # Read the new observation
        observation = self._get_obs()

        # Extract info from vision.json for rewards and achievements
        has_gathered = self.has_gathered
        is_harvesting = action[6] > 0

        # Calculate reward
        tree_dist = np.linalg.norm(observation[3:6])
        ore_dist = np.linalg.norm(observation[6:9])
        min_dist = min(tree_dist, ore_dist)
        interaction_range = 3.0

        reward = 0.0

        # Reward for getting closer to resources (+10 as requested)
        if self.prev_tree_dist is not None:
            prev_min = min(self.prev_tree_dist, self.prev_ore_dist)
            if min_dist < prev_min:
                reward += 10.0  # Movement reward
            elif min_dist > prev_min:
                reward -= 1.0   # Penalty for moving away

        # Reward for being very close (encourages parking/staying)
        if min_dist < interaction_range:
            reward += 0.5
            
        object_type = ""
        if min_dist == tree_dist:
            if 'hemp' in self.current_tree_name.lower():
                object_type = "Hemp"
            else:
                object_type = "Wood"
        elif min_dist == ore_dist:
            object_type = "Ore"

        # Major bonus for successful "Hit" (+100, which is 10x movement +10)
        # We only reward if they actually executed the harvesting action and it landed.
        if is_harvesting and has_gathered and min_dist < interaction_range:
            reward += 100.0
            
            # Since the object might disappear from the json stream, this incentivizes hitting until gone
            if object_type == "Wood":
                self.wood_count += 1
            elif object_type == "Ore":
                self.tool_count += 1  # Using tool_count to track Ore
            elif object_type == "Hemp":
                self.cloth_count += 1

        # Small step penalty to encourage efficiency
        reward -= 0.1

        # Update previous distances
        self.prev_tree_dist = tree_dist
        self.prev_ore_dist = ore_dist

        # Episode ends after max_steps
        truncated = self.steps >= self.max_steps
        terminated = False

        # Disabling old simple string-based achievements loop since it's already updated inside harvest logic.
        
        # Determine Current Goal
        health = observation[9] * 100.0
        if health < 40.0:
            goal = "Low Health - Recovering"
        elif has_gathered:
            goal = f"Successfully Harvested {observation[3:6] if tree_dist < ore_dist else observation[6:9]}"
        elif min_dist < 3.0:
            goal = f"Harvesting {'Tree' if tree_dist < ore_dist else 'Ore'}..."
        elif min_dist < 15.0:
            goal = f"Approaching {'Tree' if tree_dist < ore_dist else 'Ore'}..."
        else:
            goal = "Exploring Frontier..."

        info = {
            "current_goal": goal,
            "tree_dist": tree_dist,
            "ore_dist": ore_dist,
            "steps": self.steps,
            "has_gathered": has_gathered,
            "reward": reward,
            "wood_count": self.wood_count,
            "cloth_count": self.cloth_count,
            "is_harvesting": 1 if is_harvesting else 0,
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
        self._send_actions(np.zeros(7, dtype=np.float32))


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
