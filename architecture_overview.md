# Rust RL Architecture: Multi-Bot Parallelization

This document explains the high-performance training architecture implemented for the Rust Reinforcement Learning agent (v2_modular) using an AMD 5700 XT.

## System Overview

The system transitions from a single-agent bottleneck to an **8-way parallel stream** to maximize GPU compute saturation.

```mermaid
graph TD
    subgraph "Python Training Loop (SB3)"
        PPO[PPO Agent]
        SVE[SubprocVecEnv]
        W1[Env Worker 0]
        W2[Env Worker 1]
        W8[Env Worker 7]
    end

    subgraph "Shared Host I/O"
        V0[vision_0.json]
        A0[actions_0.json]
        V1[vision_1.json]
        A1[actions_1.json]
        V7[vision_7.json]
        A7[actions_7.json]
    end

    subgraph "Rust Game Server (Carbon v2.1)"
        BC[BotController Plugin]
        B0[Bot 0]
        B1[Bot 1]
        B7[Bot 7]
    end

    subgraph "GPU Acceleration"
        DML[DirectML / AMD 5700 XT]
    end

    %% Data Flow
    W1 <--> V0 & A0
    W2 <--> V1 & A1
    W8 <--> V7 & A7
    V0 & A0 <--> B0
    V1 & A1 <--> B1
    V7 & A7 <--> B7
    PPO <--> SVE
    SVE <--> W1 & W2 & W8
    PPO --- DML
```

## Core Components

### 1. The Carbon Plugin (`BotController.cs`)
- **Multi-Bot Management**: Spawns 8 autonomous entities (`Bot_0` to `Bot_7`) on server initialization.
- **Indexed Handshake**: Each bot listens specifically to its own action file (e.g., `actions_3.json`) and reports its vision to its own vision file (e.g., `vision_3.json`).
- **Low Latency**: Runs on a 100ms timer (10 ticks/sec), providing high-frequency updates to the agent.

### 2. The Python Environment (`environment.py`)
- **Vectorized Wrapper**: Implements the `gymnasium.Env` interface with a `bot_id` parameter.
- **Zero-Wait Step**: Removed all `time.sleep` calls to allow the Python workers to spin at maximum CPU speed during data collection.

### 3. The Training Loop (`train.py`)
- **SubprocVecEnv**: Uses Python's `multiprocessing` to run 8 environments in parallel. This bypasses the Global Interpreter Lock (GIL) and allows massive data collection speed.
- **Extreme Scaling**:
    - `batch_size = 1024`: Large matrices to saturate the AMD 5700 XT compute units.
    - `n_steps = 4096`: Collects a massive amount of data before the "GPU Update Phase."
    - `n_epochs = 30`: Forces the GPU to perform 30 passes over the data to keep power consumption and clocks high.

### 4. DirectML Integration
- Uses `torch_directml` to target the AMD Radeon RX 5700 XT.
- **VRAM Utilization**: Uses ~3.5GB of the 8GB available, leaving plenty of headroom for the high-resolution Rust server assets.

## Monitoring

- **Weights & Biases**: Real-time cloud dashboard for tracking `live_step`, `SPS`, and environmental achievements (Wood/Cloth harvested).
- **Local Dashboard**: `local_dashboard.html` provides a zero-latency view of the bots' raw JSON data directly from the `shared-data` folder.

## Project Metrics (Snapshot 2026-03-30)

| Metric | Count |
|--------|-------|
| **Git-Tracked Files** | 2,540 |
| **Total Lines of Code** | 197,032 |
| **Active Bots** | 8 |
| **GPU Target** | AMD Radeon RX 5700 XT |
