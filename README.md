# Rust RL Agent — MVP

This project trains a reinforcement-learning policy to control a bot on a private Rust dedicated server.

The current implementation is a PPO policy, not a large language model. The MVP intentionally starts with one server-side bot, structured telemetry, movement, looking, sprinting, jumping, and attacking. Crafting, building, combat intelligence, and screen-based perception are later milestones.

## Current architecture

1. Carbon spawns one or more RL_Agent entities.
2. The plugin reads actions_N.json and writes vision_N.json.
3. Python waits for a newer telemetry Tick before completing each environment step.
4. Stable-Baselines3 PPO trains a small CNN plus telemetry-vector policy.

The protocol is file based for reliability and debuggability. Both sides use atomic replacement, and every payload contains ProtocolVersion, BotId, Tick or StepId, and SessionId.

## Repository layout

- ai-agent/protocol.py — protocol constants and atomic JSON helpers
- ai-agent/environment.py — Gymnasium environment
- ai-agent/train_resnet_v2.py — repaired PPO entrypoint
- ai-agent/inference_v3.py — checkpoint inference entrypoint
- rust-plugin/BotController.cs — Carbon controller and telemetry producer
- tests/ — protocol and environment smoke tests
- start_pipeline.bat — Windows launcher for the private MVP

## Requirements

- Windows Rust dedicated server with Carbon installed
- A private or whitelisted server that you control
- Python 3.11
- Stable-Baselines3, Gymnasium, PyTorch, and the packages in requirements.txt
- DirectML is optional; CPU mode is supported for smoke tests

The server installation, shared-data directory, models, checkpoints, and virtual environment are intentionally not committed. Configure their paths locally.

## First run

From PowerShell:

    cd C:\Projects\rust-rl-agent
    py -3.11 -m venv venv
    .\venv\Scripts\python.exe -m pip install -r requirements.txt

Set the bridge variables in the same shell before starting the server:

    $env:RUST_RL_SHARED_DATA = "C:\Projects\rust-rl-agent\shared-data"
    $env:RUST_RL_BOT_COUNT = "1"
    $env:RUST_RL_NUM_ENVS = "1"
    $env:RUST_RL_INVULNERABLE = "true"

Place BotController.cs in Carbon's plugin directory, start the server, and verify that shared-data\vision_0.json is being refreshed.

Then run a short training validation:

    .\venv\Scripts\python.exe ai-agent\train_resnet_v2.py --num-envs 1 --total-timesteps 10000 --device cpu

If that completes and the telemetry values change as expected, run a longer experiment:

    .\venv\Scripts\python.exe ai-agent\train_resnet_v2.py --num-envs 1 --total-timesteps 1000000 --device auto

The generated checkpoint is models\mvp_checkpoints\rust_mvp_final.zip.

## MVP success criterion

Do not judge learning by SPS alone. The first meaningful result is:

- the bot moves toward a detected tree;
- Attack causes a confirmed inventory increase;
- wood_count and has_gathered are visible in TensorBoard;
- the behavior repeats from a new spawn position.

The plugin currently uses direct server-side movement for a controlled experiment. This is not yet equivalent to a normal client playing Rust. Keep it on a private server and do not use it to bypass anti-cheat or interact with public servers.

## Roadmap

1. One-bot telemetry and confirmed wood gathering.
2. Real collision-aware movement and explicit resource-hit events.
3. Crafting and foundation placement with confirmed server events.
4. Frame stacking or recurrent memory for navigation.
5. Additional map seeds, spawn positions, weather, and hazards.
6. Multi-bot rollout collection.
7. Optional high-level language/VLM planner above the fast low-level policy.
