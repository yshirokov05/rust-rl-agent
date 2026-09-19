# Rust RL Agent — Local Runtime Status

The repository contains the source code for the private-server MVP. The Rust server installation, Carbon installation, Python virtual environment, shared telemetry, and trained checkpoints remain local runtime assets.

Before training, verify:

- Carbon loads rust-plugin/BotController.cs.
- shared-data/vision_0.json is refreshed at roughly 10 Hz.
- Python and Carbon use the same RUST_RL_SHARED_DATA directory.
- RUST_RL_BOT_COUNT and RUST_RL_NUM_ENVS are both 1 for the first run.
- The initial short CPU run records changing wood_count, tick, and reward values.

Once the one-bot contract is proven, scale the bot count and environment count together.
