# Gemini Project Summary: Rust RL Agent

## Project Overview
This repository contains the development environment for a Rust Reinforcement Learning agent. The goal is to train an agent to interact with the game Rust via a C# plugin and a Python-based RL environment. **No game client is needed** — a server-side bot handles everything autonomously.

## Latest Changes (2026-03-17)
- **Project Recovery**: Repaired Rust Dedicated Server via SteamCMD validation.
- **Autonomous Bot**: Created `BotController.cs` — a Carbon plugin that spawns a server-side bot controlled by the Python agent via `actions.json` / `vision.json`. No Rust game client needed.
- **Bidirectional Communication**: Python writes `actions.json` → Bot moves → Bot writes `vision.json` → Python reads observations.
- **Persistent Training**: Integrated W&B (offline mode), checkpointing every 5000 steps, and automatic resume from `models/latest.zip`.
- **Reward Shaping**: Agent is rewarded with +10 for getting closer to resources (trees/ores) and +100 for a successful "Hit" (gathering).
- **W&B Integration**: Project name updated to `rust-rl-agent`. Real-time tracking of achievements like `10x Cloth` and `First Wood`.
- **Map Size**: Reduced to 1500 for faster startup (~2 min vs crash on 3000).

## Current State (PICK UP HERE)
- **Server**: NOT running. Needs to be started.
- **Plugins**: `BotController.cs` is deployed to Carbon plugins folder.
- **Agent**: `environment.py` and `train.py` are ready with SB3 PPO and Monitor wrapper.
- **Observation space**: 10 features (player XYZ, tree XYZ, ore XYZ, health).

## Step-by-Step Start Guide

### Step 1: Start the Server
```powershell
cd C:\Projects\rust-rl-agent
cmd /c ".\start_server.bat"
```
Wait ~2 minutes for `Server startup complete` in the log. The BotController will automatically spawn a bot 5 seconds after server init.

### Step 2: Verify the Bot Spawned
Check the server log file for `BotController: Bot spawned at`:
```powershell
cmd /c "Get-Content -Path C:\Projects\rust-rl-agent\server\steamapps\common\rust_dedicated\rust_server.log -Tail 10"
```

### Step 3: Start Training
```powershell
cd C:\Projects\rust-rl-agent
cmd /c ".\venv\Scripts\python ai-agent/train.py"
```
W&B will automatically log to the `rust-rl-agent` project.

### Step 4: Monitor
- W&B Dashboard: Real-time learning curves, survival time, and achievements.
- Checkpoints: `ai-agent/models/checkpoints/` (every 5,000 steps).
- Resume model: `ai-agent/models/latest_model.zip`.
- Server log: `server/steamapps/common/rust_dedicated/rust_server.log`.

## Key File Locations
| File | Path |
|------|------|
| Server Start | `C:\Projects\rust-rl-agent\start_server.bat` |
| Bot Plugin | `rust-plugin/BotController.cs` |
| Environment | `ai-agent/environment.py` |
| Training | `ai-agent/train.py` |
| Vision Data | `shared-data/vision.json` |
| Actions Data | `shared-data/actions.json` |

## Next Steps
1. Monitor W&B dashboard for learning curves.
2. Build a 2D visualizer to watch the bot on a minimap.
3. Phase 2: Add combat + base building rewards.
