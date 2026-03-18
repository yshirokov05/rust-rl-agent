# Gemini Project Summary: Rust RL Agent

## Project Overview
This repository contains the development environment for a Rust Reinforcement Learning agent. The goal is to train an agent to interact with the game Rust via a C# plugin and a Python-based RL environment. **No game client is needed** — a server-side bot handles everything autonomously.

## Latest Changes (2026-03-17)
- **Project Recovery**: Repaired Rust Dedicated Server via SteamCMD validation.
- **Autonomous Bot**: Created `BotController.cs` — a Carbon plugin that spawns a server-side bot controlled by the Python agent via `actions.json` / `vision.json`. No Rust game client needed.
- **Bidirectional Communication**: Python writes `actions.json` → Bot moves → Bot writes `vision.json` → Python reads observations.
- **Persistent Training**: Integrated W&B (offline mode), checkpointing every 5000 steps, and automatic resume from `models/latest.zip`.
- **Reward Shaping**: Agent is rewarded for approaching trees/ores and attacking when close.
- **Map Size**: Reduced to 1500 for faster startup (~2 min vs crash on 3000).

## Current State (PICK UP HERE TOMORROW)
- **Server**: NOT running. Needs to be started.
- **Plugins**: `BotController.cs` is deployed to Carbon plugins folder. `AgentEyes.cs` was removed (BotController handles vision).
- **Agent**: `environment.py` and `train.py` are ready. Dependencies installed (wandb, tensorboard, stable-baselines3).
- **Observation space**: 10 features (player XYZ, tree XYZ, ore XYZ, health).

## TOMORROW: Step-by-Step Start Guide

### Step 1: Start the Server
```powershell
cd C:\Projects\rust-rl-agent
.\start_server.bat
```
Wait ~2 minutes for `Server startup complete` in the log. The BotController will automatically spawn a bot 5 seconds after server init.

### Step 2: Verify the Bot Spawned
Check the server log file for `BotController: Bot spawned at`:
```powershell
Get-Content -Path C:\Projects\rust-rl-agent\server\steamapps\common\rust_dedicated\rust_server.log -Tail 10
```

### Step 3: Start Training
```powershell
cd C:\Projects\rust-rl-agent
.\venv\Scripts\python ai-agent/train.py
```
When W&B prompts, choose option 3 (offline) or log in with your account.

### Step 4: Monitor
- Checkpoints save to `ai-agent/models/checkpoints/`
- Resume model: `ai-agent/models/latest.zip`
- Server log: `server/steamapps/common/rust_dedicated/rust_server.log`

## Key File Locations
| File | Path |
|------|------|
| Server Start | `C:\Projects\rust-rl-agent\start_server.bat` |
| Bot Plugin | `rust-plugin/BotController.cs` |
| Vision Plugin | `rust-plugin/AgentEyes.cs` (kept as reference, NOT deployed) |
| Environment | `ai-agent/environment.py` |
| Training | `ai-agent/train.py` |
| Vision Data | `shared-data/vision.json` |
| Actions Data | `shared-data/actions.json` |

## Next Steps (After Training Starts)
1. Monitor W&B dashboard for learning curves.
2. Build a 2D visualizer to watch the bot on a minimap.
3. Scale to cloud (Azure $100 / GCP $300 student credits) for longer runs.
4. Phase 2: Add combat + base building rewards.
