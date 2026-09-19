"""Train the Rust MVP policy with Stable-Baselines3 PPO.

Start with one bot.  Set RUST_RL_NUM_ENVS above one only after the one-bot
telemetry/action contract passes the integration test on the real server.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from gymnasium import spaces

try:
    import torch_directml

    HAS_DIRECTML = True
except Exception:
    # CPU training must remain available when an optional DirectML install is
    # missing or its native runtime cannot load on this machine.
    torch_directml = None
    HAS_DIRECTML = False

try:
    from .environment import RustEnv
    from .reward_shaping import RewardShaper
except ImportError:
    from environment import RustEnv
    from reward_shaping import RewardShaper


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class RustFeaturesExtractor(BaseFeaturesExtractor):
    """Small encoder suited to the low-resolution semantic MVP input."""

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 192):
        super().__init__(observation_space, features_dim)

        self.visual = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(96, 128),
            nn.ReLU(),
        )
        self.vector = nn.Sequential(
            nn.Linear(14, 64),
            nn.ReLU(),
        )
        self._features_dim = 192

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        # normalize_images=False leaves uint8 values untouched in SB3.
        # Normalize deterministically rather than branching on image contents.
        image = observations["image"].float() / 255.0
        visual_features = self.visual(image)
        vector_features = self.vector(observations["vector"].float())
        return torch.cat((visual_features, vector_features), dim=1)


class ShapedRustEnv(RustEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shaper = RewardShaper()

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        reward += self.shaper.get_shaping_reward(observation, info)
        return observation, reward, terminated, truncated, info


class TelemetryCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.gather_events_total = 0
        self.wood_delta_total = 0.0

    @staticmethod
    def _finite_metric(info, key):
        value = info.get(key)
        if value is None:
            return None
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if np.isfinite(result) else None

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            wood_count = self._finite_metric(info, "wood_count")
            if wood_count is not None:
                self.logger.record_mean("rust/wood_count_avg", wood_count)

            if "has_gathered" in info:
                gathered = float(bool(info["has_gathered"]))
                self.logger.record_mean("rust/gathered_ratio", gathered)
                if gathered:
                    self.gather_events_total += 1

            wood_delta = self._finite_metric(info, "wood_delta")
            if wood_delta is not None:
                self.logger.record_mean("rust/wood_delta_avg", wood_delta)
                self.wood_delta_total += max(0.0, wood_delta)

            tree_distance = self._finite_metric(info, "tree_distance_m")
            if tree_distance is not None:
                self.logger.record_mean(
                    "rust/tree_distance_m_avg", tree_distance
                )

            ack_latency = self._finite_metric(info, "ack_latency_ms")
            if ack_latency is not None:
                self.logger.record_mean(
                    "rust/ack_latency_ms_avg", ack_latency
                )

            timeout_count = self._finite_metric(info, "telemetry_timeouts")
            if timeout_count is not None:
                self.logger.record_mean(
                    "rust/telemetry_timeouts_avg", timeout_count
                )

        # These monotonically increasing run totals preserve sparse events
        # even when the logger dumps only once per PPO rollout.
        self.logger.record(
            "rust/gather_events_total", self.gather_events_total
        )
        self.logger.record("rust/wood_delta_total", self.wood_delta_total)
        return True


def resolve_device(requested: str):
    requested = requested.lower()
    if requested == "cpu":
        return "cpu"

    if requested in {"auto", "directml"} and HAS_DIRECTML:
        try:
            device = torch_directml.device()
            print(f"Using DirectML device: {device}")
            return device
        except Exception as exc:
            if requested == "directml":
                raise RuntimeError(
                    f"DirectML was requested but could not initialize: {exc}"
                ) from exc

    if requested == "directml":
        raise RuntimeError(
            "DirectML was requested but torch-directml is not installed."
        )

    print("DirectML unavailable; falling back to CPU.")
    return "cpu"


def build_env(args):
    def make_env(bot_id):
        def factory():
            return ShapedRustEnv(
                bot_id=bot_id,
                shared_data_dir=args.shared_data_dir,
                observation_timeout=args.observation_timeout,
            )

        return factory

    if args.num_envs == 1:
        env = DummyVecEnv([make_env(0)])
    else:
        env = SubprocVecEnv(
            [make_env(i) for i in range(args.num_envs)],
            start_method="spawn",
        )
    return VecMonitor(env)


def parse_args():
    default_envs = int(os.environ.get("RUST_RL_NUM_ENVS", "1"))
    default_steps = int(os.environ.get("RUST_RL_TOTAL_TIMESTEPS", "1000000"))
    default_timeout = float(
        os.environ.get("RUST_RL_OBSERVATION_TIMEOUT", "10")
    )
    return argparse.ArgumentParser(
        description="Train the private Rust server MVP policy."
    ), default_envs, default_steps, default_timeout


def main():
    parser, default_envs, default_steps, default_timeout = parse_args()
    parser.add_argument("--num-envs", type=int, default=default_envs)
    parser.add_argument("--total-timesteps", type=int, default=default_steps)
    parser.add_argument("--observation-timeout", type=float, default=default_timeout)
    parser.add_argument("--shared-data-dir", default=os.environ.get("RUST_RL_SHARED_DATA"))
    parser.add_argument("--device", choices=("auto", "cpu", "directml"), default="auto")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    if args.num_envs < 1:
        parser.error("--num-envs must be at least 1")
    if args.total_timesteps < 1:
        parser.error("--total-timesteps must be positive")

    multiprocessing.freeze_support()
    torch.set_default_dtype(torch.float32)
    device = resolve_device(args.device)
    env = build_env(args)

    checkpoint_dir = Path(
        os.environ.get(
            "RUST_RL_CHECKPOINT_DIR",
            str(PROJECT_ROOT / "models" / "mvp_checkpoints"),
        )
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = PROJECT_ROOT / "runs" / "mvp"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    policy_kwargs = {
        "features_extractor_class": RustFeaturesExtractor,
        "normalize_images": False,
        "net_arch": {"pi": [128, 64], "vf": [128, 64]},
    }

    model = None
    try:
        if args.resume:
            print(f"Resuming checkpoint: {args.resume}")
            model = PPO.load(args.resume, env=env, device=device)
        else:
            model = PPO(
                "MultiInputPolicy",
                env,
                policy_kwargs=policy_kwargs,
                n_steps=256,
                batch_size=64,
                learning_rate=3e-4,
                ent_coef=0.01,
                device=device,
                verbose=1,
                tensorboard_log=str(tensorboard_dir),
            )

        checkpoint_callback = CheckpointCallback(
            save_freq=max(10000 // args.num_envs, 1),
            save_path=str(checkpoint_dir),
            name_prefix="rust_mvp",
        )
        callback = CallbackList([TelemetryCallback(), checkpoint_callback])

        print(
            f"Training {args.total_timesteps:,} timesteps with "
            f"{args.num_envs} environment(s)."
        )
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            reset_num_timesteps=args.resume is None,
        )
        final_path = checkpoint_dir / "rust_mvp_final"
        model.save(str(final_path))
        print(f"Saved final checkpoint: {final_path}.zip")
    finally:
        if model is not None and args.total_timesteps > 0:
            emergency_path = checkpoint_dir / "rust_mvp_last_exit"
            try:
                model.save(str(emergency_path))
            except Exception as exc:
                print(f"Could not save emergency checkpoint: {exc}")
        env.close()


if __name__ == "__main__":
    main()
