# Rust RL Agent — Repaired MVP Architecture

## Scope

The first milestone is a one-bot, private-server experiment. The objective is to make the environment contract observable and testable before scaling rollout collection.

## Data flow

    Carbon BotController
        |
        | atomic vision_0.json, Tick, SessionId
        v
    RustEnv.reset/step
        |
        | action_0.json, StepId, SessionId
        v
    Carbon BotController

A step is valid only when Python receives a newer Tick and the telemetry belongs to the current SessionId. Partial JSON writes and stale frames are rejected.

## Canonical action contract

    [MoveX, MoveZ, LookX, LookY, Sprint, Jump, Attack]

The values are seven float32 values in [-1, 1]. Sprint, Jump, and Attack are interpreted as boolean thresholds by the protocol. Crafting, hotbar selection, and building are intentionally not exposed until the server confirms those events.

## Observation contract

The MVP returns:

- image: 3 x 84 x 84 uint8 semantic map; zeros are allowed until semantic-map generation is added
- vector: 14 float32 values containing player position, relative nearest tree and ore positions, health, wood, stone, predator flag, and active-item ID
- telemetry info: Tick, AppliedStepId, Alive, HasGathered, inventory counts, and resource names

Nearest resource positions are relative to the bot. The plugin uses an atomic temporary-file replacement when publishing telemetry.

## Training

The default trainer uses one DummyVecEnv and a small CNN/vector encoder. Stable-Baselines3 MultiInputPolicy is used because the observation space is a Dict. Set RUST_RL_NUM_ENVS above one only after one-bot training works. SubprocVecEnv remains available for later parallel rollouts.

## What is not implemented yet

- normal client input and collision-aware locomotion
- reliable melee raycasts and explicit hit events
- crafting, inventory UI, building, upgrading, and respawn
- visual screen capture or a human-equivalent camera observation
- adversarial players and general Rust survival
- language-model planning


## Future autonomous NPC track

The long-term goal is a modular Rust companion/NPC, not one monolithic LLM policy. After the one-bot PPO bridge is proven, add server-confirmed skills behind a behavior-tree or utility planner:

1. perceive nearby entities and maintain a blackboard;
2. navigate to resources and gather them;
3. craft tools and equipment;
4. choose legal building locations and construct a base;
5. manage survival needs and threats;
6. fight NPCs or players using dedicated combat controllers.

PPO may improve individual skills such as aiming or navigation. Language-model planning is optional and belongs above these skills; it must not directly control every low-level movement tick.
