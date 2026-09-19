# Rust RL Agent — one-bot MVP

This repository trains a Stable-Baselines3 PPO policy to control a bot on a private Rust dedicated server. It is an RL controller, not an LLM. The first milestone is deliberately narrow: prove a reliable action/telemetry bridge, move one bot, and obtain a server-confirmed wood inventory increase before attempting long training runs.

The current controller uses structured server telemetry and direct server-side movement. It is not equivalent to a normal Rust client and must remain on a private or whitelisted server you control.

## Architecture

1. Carbon loads `rust-plugin/BotController.cs` and spawns the configured bot count.
2. Python atomically writes `shared-data/actions_N.json`.
3. Carbon applies the action and atomically writes `shared-data/vision_N.json`.
4. Protocol version, bot ID, session ID, tick, and applied step ID identify each transition.
5. `ai-agent/train_resnet_v2.py` trains the PPO policy after the bridge smoke gate passes.

The MVP action contract is:

```text
[MoveX, MoveZ, LookX, LookY, Sprint, Jump, Attack]
```

Crafting, building, public-server play, anti-cheat bypasses, and an LLM planner are outside this milestone.

## Repository layout

- `ai-agent/protocol.py` — canonical JSON protocol and atomic file helpers
- `ai-agent/environment.py` — Gymnasium environment
- `ai-agent/train_resnet_v2.py` — PPO training entrypoint
- `ai-agent/inference_v3.py` — checkpoint inference entrypoint
- `rust-plugin/BotController.cs` — Carbon bot controller and telemetry producer
- `scripts/bridge_smoke.py` — exact reset/forward/stop acknowledgement check
- `scripts/gather_smoke.py` — scripted tree approach and real wood-gain gate
- `tests/` — offline protocol and environment contract tests
- `start_pipeline.bat` — guarded Windows launcher

Local server files, shared telemetry, checkpoints, logs, virtual environments, and experiment artifacts are intentionally ignored by Git.

## Requirements

- Native Windows 10/11
- Python 3.11 x64
- Rust Dedicated Server with Carbon installed
- One private server instance
- CPU mode for the first smoke run

`requirements.txt` is the pinned CPU-first training environment. It excludes the old dashboard, W&B, OpenCV, and migration dependencies because none are needed by the MVP path.

DirectML is optional and public-preview software. Use the separate `requirements-directml.txt` only after CPU training works. The AMD RX 5700 XT is adequate for this small model, but the one-bot 10 Hz server loop—not GPU throughput—is expected to be the first bottleneck.

## First-time setup

Open PowerShell in the repository:

```powershell
cd C:\Projects\rust-rl-agent
py -3.11 -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m pip check
.\venv\Scripts\python.exe -m compileall -q ai-agent tests scripts\bridge_smoke.py scripts\gather_smoke.py
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

The test run must complete without skips. CI enforces at least four tests, zero skips, successful source compilation, and `pip check` on Windows/Python 3.11.

## Configure Carbon and the server

Copy the plugin into the Carbon plugin directory for the same dedicated-server installation that will be launched. A typical layout is:

```powershell
$serverRoot = "C:\Projects\rust-rl-agent\server\server\steamapps\common\rust_dedicated"
Copy-Item .\rust-plugin\BotController.cs `
  (Join-Path $serverRoot "carbon\plugins\BotController.cs") -Force
```

Set machine-specific paths in the same PowerShell session. `runtime.env.example` documents every supported variable; it is a reference file and is not loaded automatically.

```powershell
$env:RUST_DEDICATED_EXE = Join-Path $serverRoot "RustDedicated.exe"
$env:RUST_RL_BUNDLE_PATH = Join-Path $serverRoot "RustDedicated_Data\Bundles"
$env:RUST_RL_SHARED_DATA = "C:\Projects\rust-rl-agent\shared-data"
$env:RUST_RL_BOT_COUNT = "1"
$env:RUST_RL_NUM_ENVS = "1"
$env:RUST_RL_INVULNERABLE = "true"
$env:RUST_RL_TOTAL_TIMESTEPS = "10000"
$env:RUST_RL_DEVICE = "cpu"
```

Restart the server after changing environment variables or replacing the plugin. A process that is already running will not inherit new environment values.

## Guarded first run

Make sure no `RustDedicated.exe` process is already running, then launch:

```powershell
.\start_pipeline.bat
```

The launcher:

1. preserves environment overrides;
2. starts Rust with its executable directory as the working directory;
3. fixes the map seed and world size unless overridden;
4. writes the Rust server log to a timestamped `artifacts\runs\...` directory;
5. runs `scripts\bridge_smoke.py` before training;
6. requires exact acknowledgements for `Reset=true` step 0, one forward step, and a neutral stop step;
7. runs `scripts\gather_smoke.py` to approach a telemetry-selected tree and require a real `WoodCount` increase;
8. refuses to train if either gate is stale, mismatched, timed out, cannot move, or cannot gather wood;
9. starts TensorBoard and a 10,000-step CPU run only after both gates pass.

The smoke verifiers write compact JSON evidence files beside `bridge_smoke.log`, `gather_smoke.log`, `rust_server.log`, and `training.log`. A stale `vision_0.json` cannot pass because every run uses new session IDs and exact applied-step acknowledgements.

If the bridge gate fails, training does not start. The launcher leaves the server running so you can inspect `rust_server.log`; stop it manually after diagnosis.

If the server is already running intentionally, do not use the launcher. Run only the bridge gate:

```powershell
.\venv\Scripts\python.exe scripts\bridge_smoke.py `
  --shared-data-dir $env:RUST_RL_SHARED_DATA `
  --timeout 180
```

## Observe and evaluate

TensorBoard is available at `http://127.0.0.1:6006/` when enabled by the launcher. Check the timestamped run directory as well as `runs\mvp` and `models\mvp_checkpoints`.

Do not use steps per second or a completed training command as proof of learning. The first meaningful result requires all of the following:

- ticks advance and every telemetry payload acknowledges the intended session and step;
- a forward action produces a measurable position change;
- attack produces a confirmed positive wood inventory delta;
- reward, wood delta, gather success, tree distance, bridge latency, and timeouts are recorded;
- the trained policy beats a random policy on repeated server resets/new spawn positions.

At 10 Hz, one bot has a hard collection ceiling near 10 steps/second: 10,000 steps take at least about 17 minutes, 100,000 about 2.8 hours, and 1,000,000 about 27.8 hours before update overhead. Do not start the 1M run until the scripted wood-gather test and repeated-reset evaluation pass.

## Optional DirectML test

After the CPU smoke succeeds, create a separate environment for the pinned DirectML compatibility tuple. Do not install it over the CPU environment:

```powershell
py -3.11 -m venv venv-dml
.\venv-dml\Scripts\python.exe -m pip install --upgrade pip
.\venv-dml\Scripts\python.exe -m pip install -r requirements-directml.txt
.\venv-dml\Scripts\python.exe -m pip check
.\venv-dml\Scripts\python.exe -c "import torch, torch_directml; d=torch_directml.device(); x=torch.tensor([1.0]).to(d); print(torch.__version__, d, (x+x).item())"
$env:RUST_RL_PYTHON_EXE = (Resolve-Path .\venv-dml\Scripts\python.exe).Path
$env:RUST_RL_DEVICE = "directml"
```

Benchmark CPU and DirectML on the same short configuration. Keep CPU if DirectML is slower or encounters an unsupported operator. Renting cloud compute is premature until one-bot learning is repeatable; scale rollout-producing server instances first, then model-update compute.

## Longer runs

After the live action, wood-gather, reset, and evaluation gates pass:

```powershell
$env:RUST_RL_TOTAL_TIMESTEPS = "100000"
$env:RUST_RL_DEVICE = "cpu"
.\start_pipeline.bat
```

The final checkpoint is written under `models\mvp_checkpoints`. `Ctrl+C` in the training window triggers the trainer's last-exit checkpoint path.

## Long-term NPC direction

This repository currently validates a narrow PPO controller and server bridge. The intended autonomous-companion direction is modular: a behavior-tree or utility planner will eventually compose server-confirmed gathering, crafting, building, survival, and combat skills. That future NPC track is not yet implemented, and no claim of full Rust autonomy should be made until each skill passes an integration test.

## Current limitations

- semantic image telemetry is currently empty;
- movement bypasses normal client locomotion and collision behavior;
- melee hit confirmation and wood collection still require live Carbon validation;
- randomized spawn curricula, death recovery, crafting, building, and general Rust survival are later milestones;
- multi-bot rollout collection should remain disabled until the one-bot contract is repeatable.
- historical W&B run files remain tracked; the ignore rule prevents new runs, while removing old artifacts is a separate repository-cleanup change.
