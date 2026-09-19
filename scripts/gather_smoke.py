"""Scripted tree approach and wood-gather gate for the live Carbon bridge."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "ai-agent"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from bridge_smoke import compact_telemetry, integer, wait_for_exact_ack  # noqa: E402
from protocol import ACTION_SIZE, atomic_write_json, build_action_payload, resolve_shared_data_dir  # noqa: E402


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def finite_number(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise RuntimeError(f"{name} must be finite")
    return result


def tree_target(payload: dict[str, Any]) -> tuple[str, float, float, float]:
    tree = payload.get("NearestTree")
    if not isinstance(tree, dict):
        raise RuntimeError("No target tree is present in telemetry")
    position = tree.get("Position")
    if not isinstance(position, dict):
        raise RuntimeError("NearestTree is missing its bot-local Position")
    name = str(tree.get("Name") or "").strip()
    if not name:
        raise RuntimeError("NearestTree is missing its name")
    x = finite_number(position.get("X"), "NearestTree.Position.X")
    z = finite_number(position.get("Z"), "NearestTree.Position.Z")
    distance = finite_number(tree.get("Distance"), "NearestTree.Distance")
    return name, x, z, distance


def build_action(
    values: list[float], *, bot_id: int, step_id: int, session_id: str, reset: bool
) -> dict[str, Any]:
    if len(values) != ACTION_SIZE:
        raise ValueError("invalid action size")
    return build_action_payload(
        values,
        bot_id=bot_id,
        step_id=step_id,
        session_id=session_id,
        reset=reset,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Require a scripted bot to approach a tree and gain wood."
    )
    parser.add_argument("--bot-id", type=int, default=0)
    parser.add_argument("--shared-data-dir", default=None)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--poll-interval", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--attack-distance", type=float, default=3.0)
    parser.add_argument(
        "--evidence-dir",
        default=str(PROJECT_ROOT / "artifacts" / "gather-smoke"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.bot_id < 0 or args.max_steps < 1:
        raise SystemExit("bot-id must be non-negative and max-steps positive")

    shared = resolve_shared_data_dir(args.shared_data_dir)
    shared.mkdir(parents=True, exist_ok=True)
    action_path = shared / f"actions_{args.bot_id}.json"
    vision_path = shared / f"vision_{args.bot_id}.json"
    evidence_dir = Path(args.evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    session_id = str(uuid.uuid4())
    evidence: dict[str, Any] = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "bot_id": args.bot_id,
        "session_id": session_id,
        "steps": [],
        "success": False,
    }
    evidence_path = evidence_dir / (
        "gather_smoke_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_"
        + session_id[:8]
        + ".json"
    )

    tick = -1
    step_id = 0
    failure: Exception | None = None
    try:
        reset = build_action(
            [0.0] * ACTION_SIZE,
            bot_id=args.bot_id,
            step_id=0,
            session_id=session_id,
            reset=True,
        )
        atomic_write_json(action_path, reset)
        telemetry, latency = wait_for_exact_ack(
            vision_path,
            bot_id=args.bot_id,
            session_id=session_id,
            step_id=0,
            after_tick=tick,
            timeout=args.timeout,
            poll_interval=args.poll_interval,
        )
        tick = integer(telemetry, "Tick")
        initial_wood = finite_number(telemetry.get("WoodCount", 0), "WoodCount")
        evidence["initial"] = compact_telemetry(telemetry)
        evidence["reset_ack_latency_ms"] = latency

        for step_id in range(1, args.max_steps + 1):
            error = str(telemetry.get("LastError") or "").strip()
            if error:
                raise RuntimeError(f"Carbon bridge error: {error}")

            tree_name, local_x, local_z, distance = tree_target(telemetry)
            yaw_error_degrees = math.degrees(math.atan2(local_x, local_z))
            aligned = abs(yaw_error_degrees) <= 25.0
            attack = distance <= args.attack_distance and abs(yaw_error_degrees) <= 35.0
            move_z = 0.0 if attack or not aligned else 1.0
            look_x = clamp(yaw_error_degrees / 8.0, -1.0, 1.0)
            values = [0.0, move_z, look_x, 0.0, 1.0 if move_z else 0.0, 0.0, 1.0 if attack else 0.0]
            action = build_action(
                values,
                bot_id=args.bot_id,
                step_id=step_id,
                session_id=session_id,
                reset=False,
            )
            atomic_write_json(action_path, action)
            telemetry, latency = wait_for_exact_ack(
                vision_path,
                bot_id=args.bot_id,
                session_id=session_id,
                step_id=step_id,
                after_tick=tick,
                timeout=min(args.timeout, 10.0),
                poll_interval=args.poll_interval,
            )
            tick = integer(telemetry, "Tick")
            wood = finite_number(telemetry.get("WoodCount", 0), "WoodCount")
            evidence["steps"].append(
                {
                    "step_id": step_id,
                    "tree": tree_name,
                    "tree_distance_m": distance,
                    "yaw_error_degrees": yaw_error_degrees,
                    "attack": attack,
                    "wood_count": wood,
                    "ack_latency_ms": latency,
                }
            )
            if wood > initial_wood:
                evidence["wood_gained"] = wood - initial_wood
                evidence["final"] = compact_telemetry(telemetry)
                evidence["success"] = True
                break
        else:
            raise RuntimeError(
                f"No wood gained after {args.max_steps} scripted actions"
            )
    except Exception as exc:
        failure = exc
        evidence["error"] = str(exc)
    finally:
        stop_step = step_id + 1
        try:
            atomic_write_json(
                action_path,
                build_action(
                    [0.0] * ACTION_SIZE,
                    bot_id=args.bot_id,
                    step_id=stop_step,
                    session_id=session_id,
                    reset=False,
                ),
            )
        except Exception as stop_exc:
            evidence["stop_error"] = str(stop_exc)
        evidence["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(evidence_path, evidence)

    if failure is not None:
        print(f"GATHER_SMOKE_FAIL: {failure}", file=sys.stderr)
        print(f"Evidence: {evidence_path}", file=sys.stderr)
        return 1
    print(
        "GATHER_SMOKE_PASS: scripted approach/attack produced "
        f"{evidence['wood_gained']:.0f} wood"
    )
    print(f"Evidence: {evidence_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
