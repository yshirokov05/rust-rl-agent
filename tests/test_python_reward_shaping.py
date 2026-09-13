import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ai-agent"))

from reward_shaping import RewardShaper


class RewardShaperTests(unittest.TestCase):
    def setUp(self):
        self.shaper = RewardShaper()
        self.rock_observation = {
            "vector": np.array([0.0] * 13 + [1.0], dtype=np.float32)
        }
        self.forward_attack = np.array(
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            dtype=np.float32,
        )

    def test_missing_tree_cannot_earn_positive_shaping(self):
        info = {
            "tree_available": False,
            "tree_distance_m": None,
            "ore_available": True,
            "ore_distance_m": 1.0,
            "last_action": self.forward_attack,
        }
        self.assertEqual(
            self.shaper.get_shaping_reward(self.rock_observation, info),
            0.0,
        )

    def test_thresholds_are_in_meters(self):
        far_info = {
            "tree_available": True,
            "tree_distance_m": 300.0,
            "last_action": self.forward_attack,
        }
        near_info = {
            "tree_available": True,
            "tree_distance_m": 2.0,
            "last_action": self.forward_attack,
        }
        self.assertEqual(
            self.shaper.get_shaping_reward(self.rock_observation, far_info),
            0.0,
        )
        self.assertGreater(
            self.shaper.get_shaping_reward(self.rock_observation, near_info),
            0.05,
        )


if __name__ == "__main__":
    unittest.main()
