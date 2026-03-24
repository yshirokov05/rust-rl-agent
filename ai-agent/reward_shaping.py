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

    def get_milestone_bonus(self, info):
        """Check for significant events like craft completions."""
        if info.get("has_plan", False):
            return 100.0
        return 0.0
