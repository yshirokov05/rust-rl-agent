# Gemini Project Summary: Rust RL Agent

## Project Overview
This repository contains the development environment for a Rust Reinforcement Learning agent. The goal is to train an agent to interact with the game Rust via a C# plugin and a Python-based RL environment.

## Latest Changes (2026-03-17)
- **Repository Initialization**: Cloned the source and configured local git identity.
- **Environment Setup**: 
    - Created a Python virtual environment (`venv`).
    - Installed core dependencies: `stable-baselines3`, `gymnasium`, `wandb`, `opencv-python`.
    - Installed ROCm-compatible PyTorch for AMD GPU support.
- **Repository Cleanup**: Removed unrelated project files and folders to ensure the repository only contains the Rust RL project.
- **GPU Verification**: Verified PyTorch installation. Note that ROCm-compatible PyTorch was installed but hardware acceleration was not detected in the Windows environment.

## Technical Suggestions
- **AMD GPU Acceleration on Windows**: If ROCm support continues to be an issue, it is highly recommended to use **DirectML** for hardware acceleration on Windows with AMD GPUs.
    - Command: `.\venv\Scripts\pip install torch-directml`
- **Next Steps**: Proceed with configuring the `ai-agent/` environment and linking it with the `rust-plugin/` data output.
