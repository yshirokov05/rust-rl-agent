import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ai-agent"))

try:
    from environment import RustEnv
    from protocol import atomic_write_json, read_json
    ENV_IMPORT_ERROR = None
except Exception as exc:
    RustEnv = None
    atomic_write_json = None
    read_json = None
    ENV_IMPORT_ERROR = exc


@unittest.skipIf(
    ENV_IMPORT_ERROR is not None,
    f"Gymnasium environment dependencies unavailable: {ENV_IMPORT_ERROR}",
)
class EnvironmentContractTests(unittest.TestCase):
    @staticmethod
    def telemetry(tick, session_id, wood=0, gathered=False):
        return {
            "ProtocolVersion": 1,
            "BotId": 0,
            "Tick": tick,
            "AppliedStepId": tick,
            "SessionId": session_id,
            "Alive": True,
            "HasGathered": gathered,
            "Health": 100,
            "WoodCount": wood,
            "StoneCount": 0,
            "ActiveItem": "rock",
            "PlayerPosition": {"X": 0, "Y": 0, "Z": 0},
            "NearestTree": {
                "Name": "pine_a",
                "Distance": 2,
                "Position": {"X": 0, "Y": 0, "Z": 2},
            },
            "NearestOre": None,
        }

    def test_step_waits_for_new_tick(self):
        with tempfile.TemporaryDirectory() as directory:
            env = RustEnv(
                bot_id=0,
                shared_data_dir=directory,
                observation_timeout=1.0,
                poll_interval=0.002,
                episode_length_range=(10, 10),
            )
            session_id = env.session_id
            vision_path = Path(directory) / "vision_0.json"
            action_path = Path(directory) / "actions_0.json"

            atomic_write_json(
                vision_path,
                self.telemetry(0, session_id),
            )
            observation, reset_info = env.reset()
            self.assertEqual(reset_info["tick"], 0)
            self.assertEqual(observation["vector"].shape, (14,))

            def publish_next_tick():
                deadline = time.monotonic() + 0.5
                while time.monotonic() < deadline:
                    action = read_json(action_path)
                    if action is not None and action.get("StepId") == 1:
                        atomic_write_json(
                            vision_path,
                            self.telemetry(
                                1,
                                session_id,
                                wood=50,
                                gathered=True,
                            ),
                        )
                        return
                    time.sleep(0.002)

            publisher = threading.Thread(target=publish_next_tick)
            publisher.start()
            _, reward, terminated, truncated, info = env.step(
                np.zeros(7, dtype=np.float32)
            )
            publisher.join(timeout=1)

            self.assertEqual(info["tick"], 1)
            self.assertGreater(reward, 0)
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            env.close()


if __name__ == "__main__":
    unittest.main()
