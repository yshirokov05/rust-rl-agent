import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
import os
import socket

class RustEnv(gym.Env):
    """
    A Gymnasium environment for a Rust AI agent.
    Communicates with the game server via UDP for vision (lockstep sync) and JSON for actions.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, bot_id=0, actions_path=None):
        super(RustEnv, self).__init__()
        root_dir = r"C:\Projects\rust-rl-agent"
        self.actions_path = actions_path or os.path.join(root_dir, "shared-data", f"actions_{bot_id}.json")
        self.steps = 0
        self.max_steps = 2048  # Episode length
        self.prev_tree_dist = None
        self.prev_ore_dist = None
        
        self.current_tree_name = "None"
        self.current_ore_name = "None"
        # Internal Phase tracking
        self.has_plan_crafted = False
        self.has_foundation_deployed = False
        self.has_gathered = False

        # Ensure shared-data directory exists for writing actions
        os.makedirs(os.path.dirname(self.actions_path), exist_ok=True)

        # Action space: [MoveX, MoveZ, LookX, LookY, Sprint, Jump, Attack, Hotbar, Craft, Radial] = 10
        self.action_space = spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)

        # Observation space transitioning to a Multi-Input Dict: 
        # Resolves the image vector to fit ResNet requirements and the secondary data via 14D vector array.
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(3, 224, 224), dtype=np.uint8),
            "vector": spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)
        })
        
        # Diversification Config
        self.obs_noise_std = 0.05
        self.episode_length_range = (1800, 2200)

        # UDP Socket Setup
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Bind to a local port derived from bot_id
        self.port = 5000 + bot_id
        self.sock.bind(('127.0.0.1', self.port))
        # 2-second timeout to prevent deadlocks if server stops
        self.sock.settimeout(2.0)

    def _send_actions(self, action):
        """Write agent actions to actions.json for the BotController plugin."""
        # Process discrete actions from continuous Box
        hotbar_select = int(np.clip(action[7], -1, 1) * 3 + 3) # 0 to 6
        craft_action = int(action[8] > 0) # 0 or 1
        radial_menu = int(np.clip(action[9], -1, 1) * 1.5 + 1.5) # 0 to 3

        actions = {
            "MoveX": float(action[0]),
            "MoveZ": float(action[1]),
            "LookX": float(action[2]),
            "LookY": float(action[3]),
            "Sprint": bool(action[4] > 0),
            "Jump": bool(action[5] > 0),
            "Attack": bool(action[6] > 0),
            "HotbarSelect": hotbar_select,
            "CraftAction": craft_action,
            "RadialMenu": radial_menu
        }
        try:
            with open(self.actions_path, 'w') as f:
                json.dump(actions, f)
        except PermissionError:
            pass  # Plugin is reading, try next tick

    def _get_obs(self):
        """Wait for UDP vision data packet to lockstep sync with server frames."""
        try:
            data_bytes, addr = self.sock.recvfrom(65536)
            data = json.loads(data_bytes.decode('utf-8'))
        except (socket.timeout, json.JSONDecodeError):
            self.current_tree_name = "None"
            self.current_ore_name = "None"
            self.has_gathered = False
            return {"image": np.zeros((3, 224, 224), dtype=np.uint8), "vector": np.zeros(14, dtype=np.float32)}

        # Semantic Image Setup
        b64_map = data.get('SemanticMapBase64', '')
        if b64_map:
            import base64
            import cv2
            raw_bytes = base64.b64decode(b64_map)
            map_array = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((84, 84, 3))
            
            # Upscale 84x84 native resolution to ResNet (224x224) via NEAREST NEIGHBOR.
            # Interpolating categorical entities (e.g. 1=Tree, 2=Stone) must not create decimals!
            map_array = cv2.resize(map_array, (224, 224), interpolation=cv2.INTER_NEAREST)
            
            # Tensor formatting channels first
            image_obs = np.transpose(map_array, (2, 0, 1))
        else:
            image_obs = np.zeros((3, 224, 224), dtype=np.uint8)

        # 14-DIM Vector Extraction
        player = data.get('PlayerPosition') or {'X': 0, 'Y': 0, 'Z': 0}
        
        tree_data = data.get('NearestTree') or {}
        tree_pos = tree_data.get('Position', {'X': 0, 'Y': 0, 'Z': 0})
        self.current_tree_name = tree_data.get('Name') or "None"

        ore_data = data.get('NearestOre') or {}
        ore_pos = ore_data.get('Position', {'X': 0, 'Y': 0, 'Z': 0})
        self.current_ore_name = ore_data.get('Name') or "None"

        health = data.get('Health', 100.0)
        self.has_gathered = data.get('HasGathered', False)
        
        # New Noob Fields
        active_item = data.get('ActiveItem', 'none')
        item_id = 0
        if 'plan' in active_item: item_id = 1
        elif 'hammer' in active_item: item_id = 2
        elif 'rock' in active_item: item_id = 3
        
        wood = data.get('WoodCount', 0)
        stone = data.get('StoneCount', 0)
        predator = 1.0 if data.get('IsPredatorNearby', False) else 0.0

        vec_obs = np.array([
            player['X'] / 1000.0, player['Y'] / 1000.0, player['Z'] / 1000.0, # Scale by world size
            tree_pos['X'] / 100.0, tree_pos['Y'] / 100.0, tree_pos['Z'] / 100.0, # Scale local neighborhood
            ore_pos['X'] / 100.0, ore_pos['Y'] / 100.0, ore_pos['Z'] / 100.0,
            health / 100.0,
            wood / 1000.0,
            stone / 1000.0,
            predator,
            item_id / 3.0
        ], dtype=np.float32)

        # APPLY DOMAIN RANDOMIZATION
        noise = np.random.normal(0, self.obs_noise_std, size=(9,))
        vec_obs[:9] += noise

        return {"image": image_obs, "vector": vec_obs}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.max_steps = np.random.randint(self.episode_length_range[0], self.episode_length_range[1] + 1)
        
        self.prev_tree_dist = None
        self.prev_ore_dist = None
        self.has_plan_crafted = False
        self.has_foundation_deployed = False

        # Send a "stop" action 
        self._send_actions(np.zeros(10, dtype=np.float32))

        # Clear UDP buffer before getting first obs to ensure fresh frame
        self.sock.setblocking(False)
        try:
            while True:
                self.sock.recvfrom(65536)
        except BlockingIOError:
            pass
        self.sock.setblocking(True)
        self.sock.settimeout(2.0)

        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
        self.steps += 1

        # Send actions to the game server
        self._send_actions(action)

        # UDP Recv blocks until frame arrives -> Automatic 10 FPS lockstep sync!
        observation = self._get_obs()
        vec_obs = observation["vector"]

        # Multi-Phase Reward Logic
        tree_dist = np.linalg.norm(vec_obs[3:6]) if self.current_tree_name != "None" else 999.0
        ore_dist = np.linalg.norm(vec_obs[6:9]) if self.current_ore_name != "None" else 999.0
        min_dist = min(tree_dist, ore_dist)
        interaction_range = 3.0
        
        wood_count = vec_obs[10] * 1000.0
        item_id = round(vec_obs[13] * 3.0) 
        craft_action = action[8] > 0
        radial_menu = int(np.clip(action[9], -1, 1) * 1.5 + 1.5)

        reward = 0.0
        
        # Penalize predator proximity
        if vec_obs[12] > 0.5:
            reward -= 1.0

        # Phase 1: Gather Wood (Target: 50 wood)
        if wood_count < 50 and not self.has_plan_crafted:
            if self.prev_tree_dist is not None and min_dist < 900:
                # [POTENTIAL-BASED REWARD] 
                # Reward the agent for closing the distance to the target.
                # Multiplier of 10.0 gives a noticeable signal for small moves.
                dist_delta = self.prev_tree_dist - tree_dist
                reward += dist_delta * 10.0
            
            if self.has_gathered and min_dist < interaction_range:
                reward += 10.0 # Harvesting reward

        # Phase 2: Crafting the Plan & Equipping
        elif wood_count >= 50 and item_id != 1 and not self.has_plan_crafted:
            # Reward triggering craft action once we have wood
            if craft_action:
                reward += 50.0
                self.has_plan_crafted = True

        # Phase 3: Building a Foundation
        elif self.has_plan_crafted and item_id == 1 and not self.has_foundation_deployed:
            # Plan is equipped! Reward deploying a foundation
            # To deploy, they need to attack (left click) or use radial (1)
            if radial_menu == 1 and action[6] > 0:
                reward += 200.0
                self.has_foundation_deployed = True

        # Phase 4: Success, heavily reward survival/standing on foundation (TBD)
        elif self.has_foundation_deployed:
            reward += 1.0

        # Small step penalty
        reward -= 0.1
        self.prev_tree_dist = tree_dist
        self.prev_ore_dist = ore_dist

        terminated = False
        truncated = self.steps >= self.max_steps
        info = { 
            "wood_count": wood_count, 
            "has_plan": self.has_plan_crafted,
            "last_action": action
        }

        # [DIAGNOSTIC] Step Audit
        if self.steps % 100 == 0:
            print(f"[BOT_{self.port-5000}] Step: {self.steps} | TreeDist: {tree_dist:.2f} | Reward: {reward:.4f} | Action: {action[0:4]}")

        return observation, reward, terminated, truncated, info

    def render(self): pass
    def close(self): self._send_actions(np.zeros(10, dtype=np.float32))

if __name__ == "__main__":
    env = RustEnv()
    obs, info = env.reset()
    print(f"Initial Observation: {obs}")
