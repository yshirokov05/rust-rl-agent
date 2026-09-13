"""Optional auxiliary rewards for the Rust MVP policy.

The environment's primary rewards must come from confirmed server events.
This module only adds small shaping terms and never marks crafting/building
as complete based on an issued command.
"""

from __future__ import annotations

import numpy as np


class RewardShaper:
    def __init__(self, config=None):
        self.config = config or {}

    def get_shaping_reward(self, observation, info):
        reward = self._calculate_possession_reward(observation, info)
        reward += self._calculate_aggression_reward(observation, info)
        return float(reward)

    @staticmethod
    def _tree_distance_m(info):
        if not info.get("tree_available", False):
            return None
        try:
            distance = float(info["tree_distance_m"])
        except (KeyError, TypeError, ValueError):
            return None
        return distance if np.isfinite(distance) and distance >= 0.0 else None

    def _calculate_possession_reward(self, observation, info):
        vector = observation.get("vector", np.zeros(14, dtype=np.float32))
        item_id = round(float(vector[13]) * 3.0)
        tree_dist_m = self._tree_distance_m(info)

        if tree_dist_m is not None and tree_dist_m < 5.0 and item_id == 3:
            return 0.05
        return 0.0

    def _calculate_aggression_reward(self, observation, info):
        action = np.asarray(
            info.get("last_action", np.zeros(7, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1)
        if action.size < 7:
            return 0.0

        is_attacking = action[6] > 0.0
        is_moving_forward = action[1] > 0.5
        is_standing_still = np.linalg.norm(action[0:2]) < 0.1
        tree_dist_m = self._tree_distance_m(info)

        if (
            tree_dist_m is not None
            and is_attacking
            and is_moving_forward
            and tree_dist_m < 3.0
        ):
            return 0.02 * max(0.0, (3.0 - tree_dist_m) / 3.0)
        if is_attacking and is_standing_still:
            return -0.02
        return 0.0

    def get_milestone_bonus(self, info):
        # Kept for compatibility with older scripts.  Milestones are now
        # event-driven and are handled by the environment.
        return 0.0
