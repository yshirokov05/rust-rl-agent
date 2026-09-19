"""Gymnasium bridge for a Rust server-side training bot.

The environment uses one atomic JSON action file and one atomic JSON telemetry
file per bot.  A reset starts a fresh session and waits for its StepId 0 reset
acknowledgement.  A normal step completes only after Carbon reports a newer
Tick with the exact SessionId and AppliedStepId that Python sent.
"""

from __future__ import annotations

import base64
import os
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    from .protocol import (
        ACTION_SIZE,
        PROTOCOL_VERSION,
        atomic_write_json,
        build_action_payload,
        read_json,
        resolve_shared_data_dir,
    )
except ImportError:
    from protocol import (
        ACTION_SIZE,
        PROTOCOL_VERSION,
        atomic_write_json,
        build_action_payload,
        read_json,
        resolve_shared_data_dir,
    )


IMAGE_SHAPE = (3, 84, 84)
VECTOR_SIZE = 14


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _vector3(value: Any) -> tuple[float, float, float]:
    if not isinstance(value, Mapping):
        return 0.0, 0.0, 0.0
    return (
        _number(value.get("X")),
        _number(value.get("Y")),
        _number(value.get("Z")),
    )


def _nested_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _protocol_int(value: Any) -> int | None:
    """Return JSON integers while rejecting bools, strings, and fractions."""
    return value if type(value) is int else None


def _boolean(value: Any, default: bool) -> bool:
    return value if isinstance(value, bool) else default


def _resource_distance_m(
    resource: Mapping[str, Any], position: tuple[float, float, float]
) -> float | None:
    name = str(resource.get("Name") or "").strip()
    if not resource or not name or name.lower() == "none":
        return None

    distance = _number(resource.get("Distance"), default=-1.0)
    if distance >= 0.0:
        return distance
    return float(np.linalg.norm(np.asarray(position, dtype=np.float32)))


class RustEnv(gym.Env):
    """A lockstep Gymnasium environment backed by a local Rust server."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        bot_id: int = 0,
        shared_data_dir: str | os.PathLike[str] | None = None,
        observation_timeout: float | None = None,
        poll_interval: float = 0.01,
        episode_length_range: tuple[int, int] = (1800, 2200),
        obs_noise_std: float = 0.0,
    ) -> None:
        super().__init__()

        self.bot_id = int(bot_id)
        self.shared_data_dir = resolve_shared_data_dir(shared_data_dir)
        self.shared_data_dir.mkdir(parents=True, exist_ok=True)
        self.actions_path = self.shared_data_dir / f"actions_{self.bot_id}.json"
        self.vision_path = self.shared_data_dir / f"vision_{self.bot_id}.json"

        configured_timeout = os.environ.get("RUST_RL_OBSERVATION_TIMEOUT", "10")
        self.observation_timeout = (
            float(configured_timeout)
            if observation_timeout is None
            else float(observation_timeout)
        )
        self.poll_interval = max(0.001, float(poll_interval))
        self.episode_length_range = episode_length_range
        self.obs_noise_std = max(0.0, float(obs_noise_std))

        # MVP action contract:
        # [MoveX, MoveZ, LookX, LookY, Sprint, Jump, Attack]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(ACTION_SIZE,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0,
                    high=255,
                    shape=IMAGE_SHAPE,
                    dtype=np.uint8,
                ),
                "vector": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(VECTOR_SIZE,),
                    dtype=np.float32,
                ),
            }
        )

        self.session_id = str(uuid.uuid4())
        self.steps = 0
        self.max_steps = self.episode_length_range[1]
        self.last_tick = -1
        self.current_tree_name = "None"
        self.current_ore_name = "None"
        self.has_gathered = False
        self.alive = True
        self.prev_tree_dist_m: float | None = None
        self.prev_tree_name = "None"
        self.prev_wood_count = 0.0
        self.telemetry_timeouts = 0
        self._closed = False

    def _send_actions(
        self, action: np.ndarray, step_id: int, *, reset: bool = False
    ) -> None:
        payload = build_action_payload(
            action=action,
            bot_id=self.bot_id,
            step_id=step_id,
            session_id=self.session_id,
            reset=reset,
        )
        atomic_write_json(self.actions_path, payload)

    def _wait_for_telemetry(
        self,
        *,
        expected_step_id: int,
        min_tick: int | None,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + self.observation_timeout
        while time.monotonic() < deadline:
            payload = read_json(self.vision_path)
            if payload is not None:
                protocol_version = _protocol_int(payload.get("ProtocolVersion"))
                payload_bot_id = _protocol_int(payload.get("BotId"))
                applied_step_id = _protocol_int(payload.get("AppliedStepId"))
                tick = _protocol_int(payload.get("Tick"))
                payload_session = payload.get("SessionId")

                matches_contract = (
                    protocol_version == PROTOCOL_VERSION
                    and payload_bot_id == self.bot_id
                    and payload_session == self.session_id
                    and applied_step_id == expected_step_id
                    and tick is not None
                    and (min_tick is None or tick > min_tick)
                )
                if matches_contract:
                    return payload
            time.sleep(self.poll_interval)

        self.telemetry_timeouts += 1
        raise RuntimeError(
            "Rust telemetry did not arrive for bot "
            f"{self.bot_id} within {self.observation_timeout:.1f}s. "
            f"Expected {self.vision_path} with ProtocolVersion "
            f"{PROTOCOL_VERSION}, BotId {self.bot_id}, SessionId "
            f"{self.session_id}, AppliedStepId {expected_step_id}, and a "
            "newer Tick. Start Carbon and verify "
            "RUST_RL_SHARED_DATA."
        )

    def _decode_observation(
        self, data: Mapping[str, Any]
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        player = _vector3(data.get("PlayerPosition"))
        tree_data = _nested_mapping(data.get("NearestTree"))
        ore_data = _nested_mapping(data.get("NearestOre"))
        tree = _vector3(tree_data.get("Position"))
        ore = _vector3(ore_data.get("Position"))
        tree_distance_m = _resource_distance_m(tree_data, tree)
        ore_distance_m = _resource_distance_m(ore_data, ore)

        self.current_tree_name = str(tree_data.get("Name") or "None")
        self.current_ore_name = str(ore_data.get("Name") or "None")
        self.has_gathered = _boolean(data.get("HasGathered"), False)
        self.alive = _boolean(data.get("Alive"), True)

        active_item = str(data.get("ActiveItem") or "none").lower()
        if "plan" in active_item:
            item_id = 1
        elif "hammer" in active_item:
            item_id = 2
        elif "rock" in active_item:
            item_id = 3
        else:
            item_id = 0

        vector = np.array(
            [
                player[0] / 1000.0,
                player[1] / 1000.0,
                player[2] / 1000.0,
                tree[0] / 100.0,
                tree[1] / 100.0,
                tree[2] / 100.0,
                ore[0] / 100.0,
                ore[1] / 100.0,
                ore[2] / 100.0,
                _number(data.get("Health"), 100.0) / 100.0,
                _number(data.get("WoodCount")) / 1000.0,
                _number(data.get("StoneCount")) / 1000.0,
                1.0 if data.get("IsPredatorNearby", False) else 0.0,
                item_id / 3.0,
            ],
            dtype=np.float32,
        )

        # Domain randomization is opt-in.  Do not inject noise into the
        # initial MVP because it hides protocol and targeting failures.
        if self.obs_noise_std > 0:
            vector[:9] += self.np_random.normal(
                0.0, self.obs_noise_std, size=9
            ).astype(np.float32)

        image = np.zeros(IMAGE_SHAPE, dtype=np.uint8)
        encoded_map = data.get("SemanticMapBase64")
        if encoded_map:
            try:
                raw = base64.b64decode(str(encoded_map), validate=True)
                if len(raw) == 84 * 84 * 3:
                    image = np.transpose(
                        np.frombuffer(raw, dtype=np.uint8).reshape(84, 84, 3),
                        (2, 0, 1),
                    ).copy()
            except (ValueError, TypeError):
                # A malformed optional image must not corrupt telemetry.
                pass

        observation = {"image": image, "vector": vector}
        wood_delta_raw = data.get("WoodDelta")
        info = {
            "tick": _protocol_int(data.get("Tick")),
            "applied_step_id": _protocol_int(data.get("AppliedStepId")),
            "alive": self.alive,
            "has_gathered": self.has_gathered,
            "wood_count": _number(data.get("WoodCount")),
            "wood_delta": (
                None
                if wood_delta_raw is None
                else _number(wood_delta_raw)
            ),
            "stone_count": _number(data.get("StoneCount")),
            "health": _number(data.get("Health"), 100.0),
            "current_tree": self.current_tree_name,
            "current_ore": self.current_ore_name,
            "tree_available": tree_distance_m is not None,
            "tree_distance_m": tree_distance_m,
            "ore_available": ore_distance_m is not None,
            "ore_distance_m": ore_distance_m,
            "active_item": active_item,
            "telemetry_timeouts": self.telemetry_timeouts,
        }
        return observation, info

    def _raise_for_bridge_error(self, data: Mapping[str, Any]) -> None:
        error = str(data.get("LastError") or "").strip()
        if error:
            raise RuntimeError(
                f"Rust bridge rejected or failed bot {self.bot_id} action: {error}"
            )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)

        self.steps = 0
        self.max_steps = int(
            self.np_random.integers(
                self.episode_length_range[0],
                self.episode_length_range[1] + 1,
            )
        )
        self.prev_tree_dist_m = None
        self.prev_tree_name = "None"
        self._closed = False

        self.session_id = str(uuid.uuid4())
        action_sent_at = time.monotonic()
        self._send_actions(
            np.zeros(ACTION_SIZE, dtype=np.float32),
            step_id=0,
            reset=True,
        )
        data = self._wait_for_telemetry(
            expected_step_id=0,
            min_tick=None,
        )
        self._raise_for_bridge_error(data)
        observation, info = self._decode_observation(data)
        info["ack_latency_ms"] = (
            time.monotonic() - action_sent_at
        ) * 1000.0
        self.last_tick = info["tick"]
        self.prev_tree_dist_m = info["tree_distance_m"]
        self.prev_tree_name = self.current_tree_name
        self.prev_wood_count = info["wood_count"]
        return observation, info

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (ACTION_SIZE,):
            raise ValueError(
                f"Expected action shape {(ACTION_SIZE,)}, got {action.shape}"
            )

        self.steps += 1
        action_sent_at = time.monotonic()
        self._send_actions(action, step_id=self.steps)
        data = self._wait_for_telemetry(
            expected_step_id=self.steps,
            min_tick=self.last_tick,
        )
        self._raise_for_bridge_error(data)
        observation, info = self._decode_observation(data)
        info["ack_latency_ms"] = (
            time.monotonic() - action_sent_at
        ) * 1000.0
        self.last_tick = info["tick"]

        tree_dist_m = info["tree_distance_m"]
        wood_delta = max(0.0, info["wood_count"] - self.prev_wood_count)
        self.has_gathered = wood_delta > 0.0
        info["wood_delta"] = wood_delta
        info["has_gathered"] = self.has_gathered

        reward = 0.0
        same_tree = self.prev_tree_name == self.current_tree_name
        if (
            same_tree
            and self.prev_tree_dist_m is not None
            and tree_dist_m is not None
        ):
            progress_m = self.prev_tree_dist_m - tree_dist_m
            reward += float(np.clip(progress_m * 0.02, -0.1, 0.1))
        if wood_delta > 0.0:
            # WoodCount is persistent, so a delayed gather cannot disappear
            # while PPO is updating between actions.
            reward += 10.0
        if observation["vector"][12] > 0.5:
            reward -= 1.0
        reward -= 0.01

        terminated = not self.alive
        if terminated:
            reward -= 10.0
        truncated = self.steps >= self.max_steps

        info.update(
            {
                # Legacy consumers expect numeric distance fields.  The
                # explicit *_available and *_distance_m fields above remain
                # the source of truth for reward logic.
                "tree_dist": 999.0 if tree_dist_m is None else tree_dist_m,
                "ore_dist": (
                    999.0
                    if info["ore_distance_m"] is None
                    else info["ore_distance_m"]
                ),
                "last_action": action.copy(),
                "session_id": self.session_id,
            }
        )

        self.prev_tree_dist_m = tree_dist_m
        self.prev_tree_name = self.current_tree_name
        self.prev_wood_count = info["wood_count"]
        return observation, reward, terminated, truncated, info

    def render(self) -> None:
        return None

    def close(self) -> None:
        if not self._closed:
            try:
                self._send_actions(
                    np.zeros(ACTION_SIZE, dtype=np.float32),
                    step_id=self.steps,
                )
            except (OSError, ValueError):
                pass
            self._closed = True


if __name__ == "__main__":
    env = RustEnv()
    try:
        observation, info = env.reset()
        print("Initial telemetry:", info)
    finally:
        env.close()
