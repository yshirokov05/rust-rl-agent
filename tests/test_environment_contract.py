import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ai-agent"))

from environment import RustEnv
from protocol import ACTION_SIZE, atomic_write_json, read_json
from reward_shaping import RewardShaper


class EnvironmentContractTests(unittest.TestCase):
    @staticmethod
    def telemetry(
        tick,
        session_id,
        *,
        applied_step_id=0,
        protocol_version=1,
        bot_id=0,
        wood=0,
        gathered=False,
        tree_distance=2.0,
        ore_distance=None,
        last_error="",
    ):
        tree = None
        if tree_distance is not None:
            tree = {
                "Name": "pine_a",
                "Distance": tree_distance,
                "Position": {"X": 0, "Y": 0, "Z": tree_distance},
            }
        ore = None
        if ore_distance is not None:
            ore = {
                "Name": "stone-ore_a",
                "Distance": ore_distance,
                "Position": {"X": ore_distance, "Y": 0, "Z": 0},
            }
        return {
            "ProtocolVersion": protocol_version,
            "BotId": bot_id,
            "Tick": tick,
            "AppliedStepId": applied_step_id,
            "SessionId": session_id,
            "Alive": True,
            "HasGathered": gathered,
            "Health": 100,
            "WoodCount": wood,
            "StoneCount": 0,
            "ActiveItem": "rock",
            "PlayerPosition": {"X": 0, "Y": 0, "Z": 0},
            "NearestTree": tree,
            "NearestOre": ore,
            "LastError": last_error,
        }

    def start_reset_ack(self, env, tick, *, previous_session, **telemetry_kwargs):
        action_path = env.actions_path
        vision_path = env.vision_path
        result = {}

        def publish():
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                action = read_json(action_path)
                if (
                    action is not None
                    and action.get("Reset") is True
                    and action.get("StepId") == 0
                    and action.get("SessionId") != previous_session
                ):
                    result["action"] = action
                    atomic_write_json(
                        vision_path,
                        self.telemetry(
                            tick,
                            action["SessionId"],
                            applied_step_id=0,
                            **telemetry_kwargs,
                        ),
                    )
                    return
                time.sleep(0.001)
            result["error"] = "reset action was not published"

        publisher = threading.Thread(target=publish)
        publisher.start()
        return publisher, result

    def start_step_ack(self, env, step_id, tick, **telemetry_kwargs):
        action_path = env.actions_path
        vision_path = env.vision_path
        session_id = env.session_id
        result = {}

        def publish():
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                action = read_json(action_path)
                if (
                    action is not None
                    and action.get("Reset") is False
                    and action.get("StepId") == step_id
                    and action.get("SessionId") == session_id
                ):
                    result["action"] = action
                    atomic_write_json(
                        vision_path,
                        self.telemetry(
                            tick,
                            session_id,
                            applied_step_id=step_id,
                            **telemetry_kwargs,
                        ),
                    )
                    return
                time.sleep(0.001)
            result["error"] = "normal action was not published"

        publisher = threading.Thread(target=publish)
        publisher.start()
        return publisher, result

    def reset_env(self, env, tick=10, **telemetry_kwargs):
        previous_session = env.session_id
        publisher, result = self.start_reset_ack(
            env,
            tick,
            previous_session=previous_session,
            **telemetry_kwargs,
        )
        observation, info = env.reset(seed=123)
        publisher.join(timeout=1.0)
        self.assertFalse(publisher.is_alive())
        self.assertNotIn("error", result)
        self.assertNotEqual(env.session_id, previous_session)
        return observation, info, result["action"]

    def test_reset_starts_new_session_and_requires_exact_reset_ack(self):
        with tempfile.TemporaryDirectory() as directory:
            env = RustEnv(
                bot_id=0,
                shared_data_dir=directory,
                observation_timeout=1.0,
                poll_interval=0.001,
                episode_length_range=(10, 10),
            )
            old_session = env.session_id
            atomic_write_json(
                env.vision_path,
                self.telemetry(
                    999,
                    old_session,
                    applied_step_id=0,
                ),
            )

            observation, info, action = self.reset_env(env, tick=10)

            self.assertEqual(info["tick"], 10)
            self.assertEqual(info["applied_step_id"], 0)
            self.assertTrue(action["Reset"])
            self.assertEqual(action["SessionId"], env.session_id)
            self.assertTrue(env.observation_space.contains(observation))

            first_session = env.session_id
            _, second_info, _ = self.reset_env(env, tick=11)
            self.assertNotEqual(env.session_id, first_session)
            self.assertEqual(second_info["tick"], 11)
            env.close()

    def test_step_ignores_stale_tick_wrong_ack_session_version_and_bot(self):
        with tempfile.TemporaryDirectory() as directory:
            env = RustEnv(
                bot_id=0,
                shared_data_dir=directory,
                observation_timeout=1.0,
                poll_interval=0.001,
                episode_length_range=(10, 10),
            )
            self.reset_env(env, tick=10)
            session_id = env.session_id

            def publish_candidates():
                deadline = time.monotonic() + 1.0
                while time.monotonic() < deadline:
                    action = read_json(env.actions_path)
                    if action is not None and action.get("StepId") == 1:
                        break
                    time.sleep(0.001)
                else:
                    return

                candidates = [
                    self.telemetry(11, "wrong-session", applied_step_id=1),
                    self.telemetry(12, session_id, applied_step_id=0),
                    self.telemetry(10, session_id, applied_step_id=1),
                    self.telemetry(
                        13,
                        session_id,
                        applied_step_id=1,
                        protocol_version=2,
                    ),
                    self.telemetry(
                        14,
                        session_id,
                        applied_step_id=1,
                        bot_id=1,
                    ),
                    self.telemetry(
                        15,
                        session_id,
                        applied_step_id=1,
                        wood=50,
                        gathered=True,
                    ),
                ]
                for payload in candidates:
                    atomic_write_json(env.vision_path, payload)
                    time.sleep(0.015)

            publisher = threading.Thread(target=publish_candidates)
            publisher.start()
            _, reward, terminated, truncated, info = env.step(
                np.zeros(ACTION_SIZE, dtype=np.float32)
            )
            publisher.join(timeout=1.0)

            self.assertEqual(info["tick"], 15)
            self.assertEqual(info["applied_step_id"], 1)
            self.assertGreater(reward, 0.0)
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            env.close()

    def test_environment_progress_reward_is_tree_only_and_meter_based(self):
        with tempfile.TemporaryDirectory() as directory:
            env = RustEnv(
                bot_id=0,
                shared_data_dir=directory,
                observation_timeout=1.0,
                poll_interval=0.001,
                episode_length_range=(10, 10),
            )
            self.reset_env(env, tick=10, tree_distance=10.0, ore_distance=10.0)

            publisher, result = self.start_step_ack(
                env,
                1,
                11,
                tree_distance=9.0,
                ore_distance=1.0,
            )
            _, reward, _, _, info = env.step(
                np.zeros(ACTION_SIZE, dtype=np.float32)
            )
            publisher.join(timeout=1.0)
            self.assertNotIn("error", result)
            self.assertAlmostEqual(info["tree_dist"], 9.0)
            self.assertAlmostEqual(info["ore_dist"], 1.0)
            self.assertAlmostEqual(reward, 0.01, places=6)

            publisher, result = self.start_step_ack(
                env,
                2,
                12,
                tree_distance=None,
                ore_distance=0.1,
            )
            _, reward, _, _, _ = env.step(
                np.zeros(ACTION_SIZE, dtype=np.float32)
            )
            publisher.join(timeout=1.0)
            self.assertNotIn("error", result)
            self.assertAlmostEqual(reward, -0.01, places=6)
            env.close()

    def test_reward_shaper_never_rewards_missing_or_ore_only_resource(self):
        shaper = RewardShaper()
        observation = {
            "vector": np.array([0.0] * 13 + [1.0], dtype=np.float32)
        }
        forward_attack = np.array(
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            dtype=np.float32,
        )

        missing_info = {
            "tree_available": False,
            "tree_distance_m": None,
            "ore_available": True,
            "ore_distance_m": 1.0,
            "last_action": forward_attack,
        }
        self.assertEqual(
            shaper.get_shaping_reward(observation, missing_info),
            0.0,
        )

        far_tree_info = {
            "tree_available": True,
            "tree_distance_m": 300.0,
            "last_action": forward_attack,
        }
        self.assertEqual(
            shaper.get_shaping_reward(observation, far_tree_info),
            0.0,
        )

        near_tree_info = {
            "tree_available": True,
            "tree_distance_m": 2.0,
            "last_action": forward_attack,
        }
        self.assertGreater(
            shaper.get_shaping_reward(observation, near_tree_info),
            0.05,
        )

    def test_persistent_wood_count_catches_delayed_gather(self):
        with tempfile.TemporaryDirectory() as directory:
            env = RustEnv(
                bot_id=0,
                shared_data_dir=directory,
                observation_timeout=1.0,
                poll_interval=0.001,
                episode_length_range=(10, 10),
            )
            self.reset_env(env, tick=10, wood=0, gathered=False)
            publisher, result = self.start_step_ack(
                env,
                1,
                11,
                wood=25,
                gathered=False,
            )
            _, reward, _, _, info = env.step(
                np.zeros(ACTION_SIZE, dtype=np.float32)
            )
            publisher.join(timeout=1.0)
            self.assertNotIn("error", result)
            self.assertEqual(info["wood_delta"], 25.0)
            self.assertTrue(info["has_gathered"])
            self.assertGreater(reward, 9.0)
            env.close()

    def test_matching_ack_with_bridge_error_fails_the_transition(self):
        with tempfile.TemporaryDirectory() as directory:
            env = RustEnv(
                bot_id=0,
                shared_data_dir=directory,
                observation_timeout=1.0,
                poll_interval=0.001,
                episode_length_range=(10, 10),
            )
            self.reset_env(env, tick=10)
            publisher, result = self.start_step_ack(
                env,
                1,
                11,
                last_error="accepted StepId 1 but movement failed",
            )
            with self.assertRaisesRegex(RuntimeError, "movement failed"):
                env.step(np.zeros(ACTION_SIZE, dtype=np.float32))
            publisher.join(timeout=1.0)
            self.assertNotIn("error", result)
            env.close()


if __name__ == "__main__":
    unittest.main()
