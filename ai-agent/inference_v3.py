"""Run a trained Rust MVP policy on one or more private-server bots."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from stable_baselines3 import PPO

try:
    import torch_directml

    HAS_DIRECTML = True
except ImportError:
    torch_directml = None
    HAS_DIRECTML = False

try:
    from .environment import RustEnv
except ImportError:
    from environment import RustEnv


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_device(requested: str):
    if requested == "cpu":
        return "cpu"
    if HAS_DIRECTML:
        try:
            return torch_directml.device()
        except Exception:
            if requested == "directml":
                raise
    if requested == "directml":
        raise RuntimeError("DirectML requested but unavailable.")
    return "cpu"


def main():
    parser = argparse.ArgumentParser(
        description="Run a PPO checkpoint through the Rust telemetry bridge."
    )
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "RUST_RL_MODEL",
            str(PROJECT_ROOT / "models" / "mvp_checkpoints" / "rust_mvp_final.zip"),
        ),
    )
    parser.add_argument("--num-bots", type=int, default=int(os.environ.get("RUST_RL_NUM_ENVS", "1")))
    parser.add_argument("--shared-data-dir", default=os.environ.get("RUST_RL_SHARED_DATA"))
    parser.add_argument("--observation-timeout", type=float, default=10.0)
    parser.add_argument("--device", choices=("auto", "cpu", "directml"), default="auto")
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args()

    if args.num_bots < 1:
        parser.error("--num-bots must be at least 1")

    device = resolve_device(args.device)
    print(f"Loading model: {args.model}")
    model = PPO.load(args.model, device=device)

    envs = [
        RustEnv(
            bot_id=bot_id,
            shared_data_dir=args.shared_data_dir,
            observation_timeout=args.observation_timeout,
        )
        for bot_id in range(args.num_bots)
    ]
    observations = [env.reset()[0] for env in envs]
    step_count = 0

    try:
        while args.max_steps <= 0 or step_count < args.max_steps:
            for index, env in enumerate(envs):
                action, _ = model.predict(
                    observations[index],
                    deterministic=True,
                )
                observation, reward, terminated, truncated, info = env.step(action)
                observations[index] = observation

                if terminated or truncated:
                    observations[index] = env.reset()[0]

                if step_count % 100 == 0:
                    print(
                        f"bot={index} tick={info.get('tick')} "
                        f"reward={reward:.3f} wood={info.get('wood_count', 0):.0f}"
                    )
            step_count += 1
    except KeyboardInterrupt:
        print("Inference stopped.")
    finally:
        for env in envs:
            env.close()


if __name__ == "__main__":
    main()
