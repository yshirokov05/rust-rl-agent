# Gemini Project Summary: Rust RL Agent

## Project Overview
This repository contains the development environment for a Rust Reinforcement Learning agent. The goal is to train an agent to interact with the game Rust via a C# plugin and a Python-based RL environment.

## Latest Changes (2026-03-17)
- **Project Recovery**: Repaired a corrupted Rust Dedicated Server installation by validating files via SteamCMD.
- **Global Rules**: Established `global_rules.md` to grant the agent self-upgrade authority for complex tasks.
- **Persistent Training**: 
    - Integrated **Weights & Biases (W&B)** for cloud-synced metric tracking and model uploads.
    - Implemented a robust **Resume Logic** in `train.py` that automatically loads `./models/latest.zip`.
    - Added **Checkpointing** every 5,000 steps for crash recovery.
- **Hardware Agnosticism**: Standardized the training script to use `device="auto"`, supporting both local AMD GPUs (DirectML) and Cloud NVIDIA GPUs (CUDA).
- **Server Deployment**: Recreated `start_server.bat` and updated the `AgentEyes` plugin with correct relative data paths.

## Current State
- **Server**: Ready for launch via `start_server.bat`.
- **Agent**: Environment and dependencies are verified. Ready for training via `ai-agent/train.py`.
- **Infrastructure**: Local training recommended for the PVE phase; cloud credits (Azure/GCP) staged for future hardware scaling.

## Next Steps
1. **Launch Server**: Start `start_server.bat` in the project root.
2. **Execute Training**: Run `ai-agent/train.py` once the server is initialized.
3. **Monitor W&B**: Track progress via the Weights & Biases dashboard.
