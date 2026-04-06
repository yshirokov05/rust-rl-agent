# Vanguard RL: Autonomous Rust Agent

Training a PPO reinforcement learning agent to survive and navigate the open-world survival game Rust — using raw game telemetry, a custom C#/Oxide server plugin, and DirectML-accelerated PyTorch on an AMD GPU. The agent sees only what the game server sees: position vectors, inventory state, and semantic vision maps.

[![PyTorch](https://img.shields.io/badge/PyTorch-PPO%20%2B%20ResNet18-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python)](https://python.org/)
[![DirectML](https://img.shields.io/badge/DirectML-AMD%20RX%205700%20XT-ED1C24?style=flat-square)](https://github.com/microsoft/DirectML)
[![C#](https://img.shields.io/badge/C%23-Oxide%2FCarbon%20Plugin-239120?style=flat-square&logo=csharp)](https://umod.org/)
[![SB3](https://img.shields.io/badge/Stable--Baselines3-PPO-FF6F61?style=flat-square)](https://stable-baselines3.readthedocs.io/)
[![LOC](https://img.shields.io/badge/LOC-191%2C401-blue?style=flat-square)](.)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│              Python Training Process (SB3)                  │
│                                                             │
│   PPO Agent ──── SubprocVecEnv ──── [Env 0 ... Env 7]      │
│       │               │                    │                │
│   DirectML        8 parallel           gymnasium.Env        │
│   AMD 5700 XT     workers             wrappers              │
└──────────────────────────┬──────────────────────────────────┘
                           │  shared-data/ (indexed JSON files)
┌──────────────────────────▼──────────────────────────────────┐
│            Rust Dedicated Server (Carbon v2.1)              │
│                                                             │
│   BotController.cs  ──►  Bot_0 ... Bot_7 (8 entities)      │
│   AgentEyes.cs      ──►  vision_N.json  (10 FPS write)      │
│   BotController.cs  ◄──  actions_N.json (policy read)       │
└─────────────────────────────────────────────────────────────┘
```

**Data flow:** The C# Carbon plugin spawns 8 bot entities on the dedicated server. Each bot writes its vision state to an indexed JSON file every 100ms. Eight Python subprocess workers read those files, compute actions via the PPO policy, and write back action commands. The GPU aggregates rollouts from all 8 workers every `n_steps=4096` for a batch update.

---

## Key Technical Highlights

- **1.6M+ training timesteps** accumulated across multi-session training with full checkpoint persistence
- **200+ SPS** (steps per second) — 8-way parallel environment using `SubprocVecEnv`, bypassing Python's GIL via `multiprocessing`
- **ResNet18 vision extractor** processes semantic map inputs from the game server; feature vectors feed directly into the PPO policy head
- **DirectML GPU acceleration** — targets AMD Radeon RX 5700 XT (8GB VRAM) via `torch_directml`; GPU config: `batch_size=1024`, `n_steps=4096`, `n_epochs=30` (~3.5GB VRAM utilization)
- **C#/Python indexed handshake** — custom Oxide/Carbon plugin handles bot spawning, NavMesh navigation, and per-bot vision/action file exchange at 10 ticks/second with zero inter-worker collision
- **191,401 lines of code** across 2,541 tracked files spanning the C# plugin, Python training stack, and server configuration
- **Weights & Biases integration** — real-time cloud tracking of `live_step`, SPS, and resource harvesting achievements (Wood/Cloth collected per episode)
- **Custom reward shaping** — `RewardShaper` class with heuristics for resource centering, tool usage (Rock/Hammer hold bonus), and survival duration scaling

---

## Project Structure

```
rust-rl-agent/
├── ai-agent/               # Python training stack
│   ├── train.py            # PPO training loop (SubprocVecEnv, DirectML)
│   ├── environment.py      # gymnasium.Env wrapper with bot_id indexing
│   ├── bc_nature_cnn.py    # ResNet18 vision feature extractor
│   └── dashboard.py        # Local training monitor
├── server/                 # Rust dedicated server installation
├── rust-plugin/            # C# Carbon plugins
│   ├── BotController.cs    # 8-bot spawner + action executor
│   └── AgentEyes.cs        # Vision broadcaster (position, inventory, heading)
├── shared-data/            # Live vision_N.json / actions_N.json exchange
├── models/                 # Saved PPO checkpoints
├── scripts/debug/          # 51 diagnostic and utility scripts
├── start_pipeline.bat      # One-command pipeline launcher
├── requirements.txt
└── architecture_overview.md
```

---

## Training Pipeline

```bash
# 1. Start the Rust dedicated server (Carbon loads BotController + AgentEyes automatically)
# 2. Launch the full Python training stack
start_pipeline.bat

# Or manually:
cd ai-agent
python train.py
```

### Requirements
```bash
pip install -r requirements.txt
# Requires: torch, torch_directml, stable-baselines3, gymnasium, wandb
```

AMD GPU required for DirectML. For CUDA, swap `torch_directml` with standard `torch` CUDA build and update the device string in `train.py`.

---

## Training Metrics (as of 2026-03-30)

| Metric | Value |
|---|---|
| Total timesteps trained | 1,600,000+ |
| Peak SPS | ~200 steps/sec |
| Parallel environments | 8 (SubprocVecEnv) |
| Batch size | 1,024 |
| n_steps per rollout | 4,096 |
| GPU | AMD RX 5700 XT (DirectML) |
| VRAM usage | ~3.5 GB / 8 GB |
| Tracked files | 2,541 |

---

## Roadmap

- [x] **Phase 1** — C#/Python bridge, NavMesh baking, multi-worker parallelization
- [x] **Phase 2** — Training stability, consistent survival >2,000 steps per episode
- [x] **Phase 3** — ResNet18 vision, reward shaping, resource harvesting behaviors
- [ ] **Phase 4** — LSTM memory for base building, combat mechanics, cloud-scale training

---

## What I Learned

**Bridging two runtimes is harder than bridging two languages.** The C#/Python file handshake looks simple but required careful index isolation, tick-rate alignment, and write-ordering awareness. A race condition across 8 concurrent bots is enough to corrupt an entire training run silently.

**The GIL is a real RL bottleneck.** Switching from `DummyVecEnv` to `SubprocVecEnv` was the single largest performance gain — turning a CPU-bound sequential loop into a genuinely parallel data collection pipeline.

**GPU utilization requires co-tuning.** Getting the 5700 XT to stay above 60% sustained utilization required jointly tuning `batch_size`, `n_steps`, and `n_epochs`. The GPU update phase needs to be computationally heavier than data collection, or you're just waiting on disk I/O.

**RL in a real game engine has no safety rails.** Unit tests don't catch server crashes, NavMesh baking failures, or file desync. Operational rigor — checkpointing, structured logging, and a live monitoring dashboard — is what separates a completed training run from a wasted night.

---

**Author:** Yury Shirokov | UC Berkeley Economics + Data Science
