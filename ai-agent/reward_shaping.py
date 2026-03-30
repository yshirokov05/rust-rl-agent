import numpy as np

class RewardShaper:
    """
    Modular class for injecting custom auxiliary rewards into the Rust RL environment.
    This class will be integrated into the step() function of RustEnv in future updates.
    """
    def __init__(self, config=None):
        self.config = config or {}
        # Sensitivity thresholds
        self.centering_threshold = 0.1 # Max offset for full centering reward
        
    def get_shaping_reward(self, observation, info):
        """
        Calculates the total auxiliary reward based on the current state.
        
        Args:
            observation (dict): The current environment observation (image, vector).
            info (dict): Additional environment info.
            
        Returns:
            float: The total shaped reward.
        """
        reward = 0.0
        
        # 1. Centering Reward (Incentivize looking directly at targets)
        reward += self._calculate_centering_reward(observation)
        
        # 2. Possession Reward (Incentivize holding specific tools)
        reward += self._calculate_possession_reward(observation)
        
        # 3. Aggression Reward (Incentivize swinging at trees)
        reward += self._calculate_aggression_reward(observation, info)
        
        # 3. Action Efficiency Reward (Incentivize minimize aimless movement)
        # To be implemented with action history
        
        return reward

    def _calculate_centering_reward(self, observation):
        """
        Reward the agent for keeping the nearest interactable object in the center of the crosshair.
        Uses the vector observation [3:9] for tree and ore relative positions.
        """
        vec = observation.get("vector", np.zeros(14))
        # Hypothetical: If LookX/LookY are included in observation, or 
        # relative positions in vector are used.
        # Currently, vector looks like: [PX, PY, PZ, TX, TY, TZ, OX, OY, OZ, ...]
        # We want to minimize the angular offset from the view vector (forward).
        return 0.0 # Placeholder for advanced spatial math

    def _calculate_possession_reward(self, observation):
        """
        Reward the agent for having a rock or tool equipped when near a node.
        """
        vec = observation.get("vector", np.zeros(14))
        item_id = round(vec[13] * 3.0) # 1=Plan, 2=Hammer, 3=Rock
        
        # Reward holding Rock (3) when near a tree or ore
        tree_dist = np.linalg.norm(vec[3:6])
        ore_dist = np.linalg.norm(vec[6:9])
        
        if (tree_dist < 5.0 or ore_dist < 5.0) and item_id == 3:
            return 0.5
        return 0.0

    def _calculate_aggression_reward(self, observation, info):
        """
        Reward the agent for triggering the Attack action while near a tree.
        Tightened in V3.1 to prevent 'air-swing' reward hacking.
        """
        last_action = info.get("last_action", np.zeros(10))
        is_attacking = last_action[6] > 0
        is_moving_forward = last_action[1] > 0.5
        is_standing_still = np.linalg.norm(last_action[0:2]) < 0.1
        
        vec = observation.get("vector", np.zeros(14))
        tree_dist = np.linalg.norm(vec[3:6])
        
        # 1. Aggression Reward (Soften in V3.2: Gradient instead of binary)
        if is_attacking and is_moving_forward:
            if tree_dist < 3.0:
                # Max 0.1 reward when touching, decaying to 0 at 3.0m
                return 0.1 * (3.0 - tree_dist) / 3.0
            
        # 2. Loitering Penalty (V3.1: Penalize stationary attacking)
        if is_attacking and is_standing_still:
            return -0.05
            
        return 0.0

    def get_milestone_bonus(self, info):
        """Check for significant events like craft completions."""
        if info.get("has_plan", False):
            return 100.0
        return 0.0
