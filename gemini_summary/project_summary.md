# Project Summary: Rust RL Agent

## Overview
This project involves building a Reinforcement Learning (RL) agent capable of playing the game **Rust** (via Steam). The system uses a C# plugin for data extraction and a Python-based Gymnasium environment for training.

## Component Breakdown

### 1. C# Plugin (`rust-plugin/AgentEyes.cs`)
- **Framework**: Carbon.
- **Purpose**: Periodically (every 100ms) scans the server environment for the nearest **Trees** and **Ores**.
- **Data Export**: Calculates relative $(x, y, z)$ vectors from the player and writes them to `shared-data/vision.json`.

### 2. Python Environment (`ai-agent/environment.py`)
- **Library**: `gymnasium`.
- **Purpose**: A standard RL environment that reads `vision.json` in its `step()` function.
- **Observation Space**: 9-dimensional vector including player position, nearest tree position, and nearest ore position (all relative where applicable).

### 3. Shared Data (`shared-data/vision.json`)
- **Purpose**: Acts as a bridge between the Rust game engine (C#) and the RL agent (Python).

## Curriculum Learning Plan
The project follows a three-phase learning strategy:
1.  **PVE**: Survival and resource gathering.
2.  **Softcore**: Basic base building and NPC combat.
3.  **PVP**: Advanced raiding and player combat.

## Current Status
- Project scaffolded with initial C# and Python files.
- Git repository initialized and pushed to [GitHub](https://github.com/yshirokov05/rust-rl-agent).
- Public repository enabled for easy scaling to cloud compute or external GPU clusters.
