"""Exercise the Rust/Carbon file bridge before starting PPO training.

The smoke sequence uses a unique session, requests a reset at StepId 0,
requires exact action acknowledgements, applies one forward action, and then
sends a neutral stop action.  A compact JSON evidence file is written whether
the check succeeds or fails.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "ai-agent"))

from protocol import (  # noqa: E402
    PROTOCOL_VERSION,
    atomic_write_json,
    build_action_payload,
    read_json,
    resolve_shared_data_dir,
)


EVIDENCE_FIELDS = (
    "ProtocolVersion",
    "BotId",
    "Tick",
    "AppliedStepId",
    "SessionId",
    "Alive",
    "HasGathered",
    "Health",
    "WoodCount",
    "StoneCount",
    "PlayerPosition",
    "PlayerYaw",
    "PlayerPitch",
    "NearestTree",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def compact_telemetry(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: payload.get(key) for key in EVIDENCE_FIELDS if key in payload}


def integer(payload: dict[str, Any], key: str, fallback: int = -1) -> int:
    try:
        return int(payload.get(key, fallback))
    except (TypeError, ValueError):
        return fallback


def position(payload: dict[str, Any]) -> tuple[float, float, float]:
    raw = payload.get("PlayerPosition")
    if not isinstance(raw, dict):
        raise RuntimeError("Telemetry is missing PlayerPosition.")
    try:
        values = tuple(float(raw[key]) for key in ("X", "Y", "Z"))
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("PlayerPosition must contain numeric X/Y/Z values.") from exc
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("PlayerPosition contains a non-finite value.")
    return values


def wait_for_exact_ack(
    vision_path: Path,
    *,
    bot_id: int,
    session_id: str,
    step_id: int,
    after_tick: int,
    timeout: float,
    poll_interval: float,
) -> tuple[dict[str, Any], float]:
    started = time.monotonic()
    deadline = started + timeout
    last_seen: dict[str, Any] | None = None

    while time.monotonic() < deadline:
        payload = read_json(vision_path)
        if payload is not None:
            last_seen = payload
            if (
                integer(payload, "ProtocolVersion") == PROTOCOL_VERSION
                and integer(payload, "BotId") == bot_id
                and str(payload.get("SessionId", "")) == session_id
                and integer(payload, "AppliedStepId") == step_id
                and integer(payload, "Tick") > after_tick
            ):
                elapsed_ms = (time.monotonic() - started) * 1000.0
                return payload, elapsed_ms
        time.sleep(poll_interval)

    summary = compact_telemetry(last_seen) if last_seen else None
    raise RuntimeError(
        f"No exact acknowledgement for bot={bot_id}, step={step_id}, "
        f"session={session_id} within {timeout:.1f}s; last telemetry={summary}"
    )


def action_payload(
    values: list[float],
    *,
    bot_id: int,
    step_id: int,
    session_id: str,
    reset: bool,
) -> dict[str, Any]:
    return build_action_payload(
        values,
        bot_id=bot_id,
        step_id=step_id,
        session_id=session_id,
        reset=reset,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the live Carbon bridge.")
    parser.add_argument("--bot-id", type=int, default=0)
    parser.add_argument("--shared-data-dir", default=os.environ.get("RUST_RL_SHARED_DATA"))
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--poll-interval", type=float, default=0.01)
    parser.add_argument("--min-movement", type=float, default=0.05)
    parser.add_argument(
        "--evidence-dir",
        default=str(PROJECT_ROOT / "artifacts" / "bridge-smoke"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.bot_id < 0:
        raise SystemExit("--bot-id must be non-negative")
    if args.timeout <= 0 or args.poll_interval <= 0:
        raise SystemExit("timeouts must be positive")

    shared_data_dir = resolve_shared_data_dir(args.shared_data_dir)
    shared_data_dir.mkdir(parents=True, exist_ok=True)
    action_path = shared_data_dir / f"actions_{args.bot_id}.json"
    vision_path = shared_data_dir / f"vision_{args.bot_id}.json"
    evidence_dir = Path(args.evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    session_id = str(uuid.uuid4())
    baseline = read_json(vision_path)
    baseline_tick = integer(baseline, "Tick") if baseline else -1
    evidence: dict[str, Any] = {
        "started_at_utc": utc_now(),
        "protocol_version": PROTOCOL_VERSION,
        "bot_id": args.bot_id,
        "session_id": session_id,
        "shared_data_dir": str(shared_data_dir.resolve()),
        "baseline": compact_telemetry(baseline) if baseline else None,
        "steps": [],
        "success": False,
    }
    evidence_path = evidence_dir / (
        "bridge_smoke_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_"
        + session_id[:8]
        + ".json"
    )

    last_tick = baseline_tick
    next_stop_step = 1
    failure: Exception | None = None
    try:
        reset_action = action_payload(
            [0.0] * 7,
            bot_id=args.bot_id,
            step_id=0,
            session_id=session_id,
            reset=True,
        )
        atomic_write_json(action_path, reset_action)
        reset_telemetry, reset_latency = wait_for_exact_ack(
            vision_path,
            bot_id=args.bot_id,
            session_id=session_id,
            step_id=0,
            after_tick=last_tick,
            timeout=args.timeout,
            poll_interval=args.poll_interval,
        )
        last_tick = integer(reset_telemetry, "Tick")
        start_position = position(reset_telemetry)
        evidence["steps"].append(
            {
                "name": "reset",
                "action": reset_action,
                "ack_latency_ms": reset_latency,
                "telemetry": compact_telemetry(reset_telemetry),
            }
        )

        forward_action = action_payload(
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            bot_id=args.bot_id,
            step_id=1,
            session_id=session_id,
            reset=False,
        )
        next_stop_step = 2
        atomic_write_json(action_path, forward_action)
        forward_telemetry, forward_latency = wait_for_exact_ack(
            vision_path,
            bot_id=args.bot_id,
            session_id=session_id,
            step_id=1,
            after_tick=last_tick,
            timeout=args.timeout,
            poll_interval=args.poll_interval,
        )
        last_tick = integer(forward_telemetry, "Tick")
        end_position = position(forward_telemetry)
        movement = math.dist(start_position, end_position)
        evidence["steps"].append(
            {
                "name": "forward",
                "action": forward_action,
                "ack_latency_ms": forward_latency,
                "telemetry": compact_telemetry(forward_telemetry),
            }
        )
        evidence["movement_m"] = movement
        if movement < args.min_movement:
            raise RuntimeError(
                f"Forward action moved only {movement:.4f}m; "
                f"expected at least {args.min_movement:.4f}m."
            )
    except Exception as exc:  # The finally block still neutralizes the bot.
        failure = exc
    finally:
        stop_action = action_payload(
            [0.0] * 7,
            bot_id=args.bot_id,
            step_id=next_stop_step,
            session_id=session_id,
            reset=False,
        )
        try:
            atomic_write_json(action_path, stop_action)
            stop_telemetry, stop_latency = wait_for_exact_ack(
                vision_path,
                bot_id=args.bot_id,
                session_id=session_id,
                step_id=next_stop_step,
                after_tick=last_tick,
                timeout=min(args.timeout, 10.0),
                poll_interval=args.poll_interval,
            )
            evidence["steps"].append(
                {
                    "name": "stop",
                    "action": stop_action,
                    "ack_latency_ms": stop_latency,
                    "telemetry": compact_telemetry(stop_telemetry),
                }
            )
        except Exception as stop_exc:
            evidence["stop_error"] = str(stop_exc)
            if failure is None:
                failure = stop_exc

    evidence["finished_at_utc"] = utc_now()
    if failure is None:
        evidence["success"] = True
    else:
        evidence["error"] = str(failure)
    atomic_write_json(evidence_path, evidence)

    if failure is not None:
        print(f"BRIDGE_SMOKE_FAIL: {failure}", file=sys.stderr)
        print(f"Evidence: {evidence_path}", file=sys.stderr)
        return 1

    print(
        "BRIDGE_SMOKE_PASS: exact reset/forward/stop acknowledgements; "
        f"movement={evidence['movement_m']:.3f}m"
    )
    print(f"Evidence: {evidence_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
