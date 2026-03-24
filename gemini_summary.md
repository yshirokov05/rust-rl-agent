# Gemini Project Summary: Rust RL Agent

## Project Overview
This repository contains the development environment for a Rust Reinforcement Learning agent. The goal is trained an agent to interact with the game Rust via a C# plugin and a Python-based RL environment. **No game client is needed** — a server-side bot handles everything autonomously.

## Latest Changes (2026-03-19)
- **Multi-Bot v0.2.1**: Scaled to 8 parallel bots in a single server instance to saturate AMD 5700 XT compute cores.
- **Hardware Saturation**: Implemented `SubprocVecEnv` (8 workers) and Extreme Scaling (`batch_size=1024`, `n_epochs=30`).
- **W&B Sync Fix**: Restored `EMERGENCY_DASHBOARD` prefix for all telemetry, resolving the dashboard stall.
- **Architecture Overview**: Created `architecture_overview.md` with Mermaid diagrams explaining the indexed JSON handshake.
- **Map Optimization**: Migrated to `WorldSize 1000` for 100% reliable initialization under 16GB RAM constraints.

## Current State (PICK UP HERE)
- **Status**: TRAINING ACTIVE (Step 44,841+).
- **Environment**: Rust Dedicated Server (Identity: v2_modular, Port: 28015).
- **RCON**: Active on Port 28016.
- **Bots**: 8 Bots active (`Bot_0` to `Bot_7`).
- **GPU**: AMD Radeon RX 5700 XT (DirectML) under active compute load.

## Step-by-Step Start Guide

### Step 1: Start the Rust Server (Background)
Ensure the server is running with the `v2_modular` identity and `server.worldsize 1000`.

### Step 2: Start Training
```powershell
cd C:\Projects\rust-rl-agent
cmd /c "venv\Scripts\python ai-agent/train.py"
```

### Step 3: Monitor
- **W&B Dashboard**: Look for `MultiBot_Extreme_Saturation` run.
- **Local Dashboard**: `local_dashboard.html` for raw JSON vitals.
- **GPU**: AMD Adrenaline overlay should show clock spikes every ~20 seconds.

## Key File Locations
| Component | Path |
|------|------|
| Multi-Bot Plugin | `rust-plugin/BotController.cs` |
| Vectorized Env | `ai-agent/environment.py` |
| Training Loop | `ai-agent/train.py` |
| Architecture Doc | `architecture_overview.md` |
| Shared Data | `shared-data/vision_{id}.json` |
