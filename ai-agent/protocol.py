"""Shared protocol primitives for the Rust server bridge.

The bridge is intentionally small and explicit:
- Python writes one complete action payload atomically.
- Carbon writes one complete telemetry payload atomically.
- Tick and session identifiers prevent stale or partial observations from
  being treated as fresh environment transitions.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


PROTOCOL_VERSION = 1
ACTION_FIELDS = (
    "MoveX",
    "MoveZ",
    "LookX",
    "LookY",
    "Sprint",
    "Jump",
    "Attack",
)
ACTION_SIZE = len(ACTION_FIELDS)


def resolve_shared_data_dir(shared_data_dir: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the shared-data directory without depending on a machine path."""
    if shared_data_dir is not None:
        return Path(shared_data_dir).expanduser()

    configured = os.environ.get("RUST_RL_SHARED_DATA")
    if configured:
        return Path(configured).expanduser()

    return Path(__file__).resolve().parent.parent / "shared-data"


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def build_action_payload(
    action: Sequence[Any],
    bot_id: int,
    step_id: int,
    session_id: str,
    *,
    reset: bool = False,
) -> dict[str, Any]:
    """Convert a policy action into the canonical JSON action payload."""
    values = list(action)
    if len(values) != ACTION_SIZE:
        raise ValueError(
            f"Expected {ACTION_SIZE} action values, received {len(values)}"
        )

    numeric = [_finite_float(value) for value in values]
    return {
        "ProtocolVersion": PROTOCOL_VERSION,
        "BotId": int(bot_id),
        "StepId": int(step_id),
        "SessionId": str(session_id),
        "Reset": bool(reset),
        "MoveX": max(-1.0, min(1.0, numeric[0])),
        "MoveZ": max(-1.0, min(1.0, numeric[1])),
        "LookX": max(-1.0, min(1.0, numeric[2])),
        "LookY": max(-1.0, min(1.0, numeric[3])),
        "Sprint": numeric[4] > 0.0,
        "Jump": numeric[5] > 0.0,
        "Attack": numeric[6] > 0.0,
    }


def atomic_write_json(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> None:
    """Write JSON via a same-directory temporary file and atomic replacement."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                dict(payload),
                handle,
                separators=(",", ":"),
                allow_nan=False,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def read_json(path: str | os.PathLike[str]) -> dict[str, Any] | None:
    """Read a JSON object, returning None for missing/partial invalid files."""
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (FileNotFoundError, PermissionError, OSError, json.JSONDecodeError):
        return None

    return value if isinstance(value, dict) else None
