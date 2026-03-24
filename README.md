# 🤖 Vanguard RL: Rust AI Agent

A state-of-the-art Reinforcement Learning pipeline designed to train autonomous agents in the high-stakes environment of Rust. The system utilizes raw screen pixels and semantic vision extraction to navigate, survive, and eventually master the game's mechanics.

## 🏗️ Architecture Stack

-   **Game Instrumentation (C#/Oxide)**: Custom Oxide plugins ([AgentEyes.cs](file:///c:/Projects/rust-rl-agent/server/server/steamapps/common/rust_dedicated/carbon/plugins/AgentEyes.cs)) broadcast real-time game data via UDP, including player position, vision vectors, and environmental feedback.
-   **RL Backbone (Python/PyTorch)**: Leverages Stable-Baselines3 PPO (Proximal Policy Optimization) with a **ResNet18** high-resolution vision extractor to process semantic maps and vector telemetry.
-   **Hardware Acceleration**: Optimized for AMD GPU performance using **DirectML** for fast tensor computations.
-   **Data Pipe**: Low-latency UDP handshake ensuring a 10 FPS lockstep sync between the Rust server and the training workers.

## 🗺️ The A-Z Roadmap

### 🏁 Phase 1: Foundation (Completed)
-   Established engine-level instrumentation and UDP vision handshake.
-   Resolved critical C# NavMesh baking race conditions.
-   Implemented multi-worker parallel environment scaling.

### 🚀 Phase 2: Stabilization (Completed)
-   Local Single-Node Training stabilized.
-   6 parallel workers achieved consistent survival (>2000 steps).
-   PPO rollout buffers and UDP port integrity verified.

### 🎯 Phase 3: Utility (Active)
-   **Reward Shaping**: Custom heuristics injected via `RewardShaper` class.
-   **Heuristic Logic**: +Points for centering resources and holding tools (Rock/Hammer).
-   **V2 Architecture**: Active 10,000-step checkpointing enabled.

### 🔮 Phase 4: Mastery (Future)
-   **Strategic Depth**: Integrating LSTM/Memory for long-term navigation and base building.
-   **Combat Mechanics**: Training for defensive and offensive interactions with environmental hazards (predators).
-   **Cloud Scaling**: Migrating to distributed cloud infrastructure for massive parallelization.

---
*Note: This project is under active development. Training metrics and SITREPs are monitored continuously by the Antigravity AI Analyst. No code was altered nor processes interrupted during this audit.*