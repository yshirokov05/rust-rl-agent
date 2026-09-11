import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ai-agent"))

from protocol import (
    ACTION_SIZE,
    atomic_write_json,
    build_action_payload,
    read_json,
)


class ProtocolTests(unittest.TestCase):
    def test_action_payload_is_canonical(self):
        payload = build_action_payload(
            [0.1, -0.2, 0.3, -0.4, 1.0, -1.0, 1.0],
            bot_id=2,
            step_id=17,
            session_id="session-a",
        )
        self.assertEqual(ACTION_SIZE, 7)
        self.assertEqual(payload["ProtocolVersion"], 1)
        self.assertEqual(payload["BotId"], 2)
        self.assertEqual(payload["StepId"], 17)
        self.assertEqual(payload["SessionId"], "session-a")
        self.assertTrue(payload["Sprint"])
        self.assertFalse(payload["Jump"])
        self.assertTrue(payload["Attack"])

    def test_wrong_action_size_is_rejected(self):
        with self.assertRaises(ValueError):
            build_action_payload(
                [0.0] * 6,
                bot_id=0,
                step_id=0,
                session_id="session-a",
            )

    def test_atomic_json_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payload.json"
            atomic_write_json(path, {"Tick": 3, "Alive": True})
            self.assertEqual(
                read_json(path),
                {"Tick": 3, "Alive": True},
            )


if __name__ == "__main__":
    unittest.main()
