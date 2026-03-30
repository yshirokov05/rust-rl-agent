# Gemini Project Summary: Rust RL Agent

## Project Overview
This repository contains the development environment for a Rust Reinforcement Learning agent. The goal is trained an agent to interact with the game Rust via a C# plugin and a Python-based RL environment. **No game client is needed** — a server-side bot handles everything autonomously.

## Latest Changes (2026-03-26 - v0.3.2)
- **Surgical Repair (v0.3.2)**: Refactored `BotController.cs` to fully adhere to Manganese Oxide architecture, specifically by removing all direct references to the deprecated `.movement` property and implementing bitwise flag setters (set_onGround, etc.).
- **Anti-Hack Bypass**: Integrated temporary `IsAdmin` flag assignment for bots during training to prevent server-side rubberbanding and allow for non-human movement patterns.
- **Vision Restoration**: Resolved a 999.0 distance default in target acquisition, restoring the agent's ability to engage with nearby trees and resources.
- **Multi-Bot v0.2.1**: Scaled to 8 parallel bots in a single server instance to saturate AMD 5700 XT compute cores.
- **Architecture Overview**: Created `architecture_overview.md` with Mermaid diagrams explaining the indexed JSON handshake.

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
