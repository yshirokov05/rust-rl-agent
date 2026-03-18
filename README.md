# Rust Reinforcement Learning Agent

This project aims to train an AI agent to play the video game **Rust** through Steam using Reinforcement Learning.

## Project Structure

- `rust-plugin/`: Contains the C# Carbon source code (`AgentEyes.cs`) used to extract in-game entity data.
- `ai-agent/`: Contains the Python Gymnasium environment (`environment.py`) for training the RL models.
- `shared-data/`: A landing zone for `vision.json`, facilitating data exchange between the game and the agent.

## Getting Started

1.  **Rust Plugin**: Install [Carbon](https://carbon.rust.community/) and copy `rust-plugin/AgentEyes.cs` to your server's plugins directory.
2.  **AI Agent**: Install dependencies (`gymnasium`, `numpy`).
3.  **Data Exchange**: Ensure both the Rust server and the Python script have read/write access to the `shared-data/` directory.

## Curriculum Learning Plan

The training process is divided into three main phases to gradually increase task complexity:

### 1. PVE (Environment Survival)
*   **Goal**: Basic survival and resource gathering.
*   **Tasks**: Locating trees/ores, harvesting resources, navigating terrain, and managing vitals.
*   **Reward**: Positive reinforcement for harvesting and staying alive.

### 2. Softcore (Intermediate Mechanics)
*   **Goal**: Basic combat and base building.
*   **Tasks**: Fighting NPCs (e.g., scientists, animals), building a simple 1x2 base, and using tools/weapons.
*   **Reward**: Positive reinforcement for successful base building and NPC takedowns.

### 3. PVP (Advanced Competition)
*   **Goal**: Advanced combat and raiding.
*   **Tasks**: Engaging with high-tier weapons, tactical positioning, and raiding enemy bases.
*   **Reward**: High rewards for player kills and successful raids.

## Data Schema (`vision.json`)

```json
{
  "PlayerPosition": { "X": 0.0, "Y": 0.0, "Z": 0.0 },
  "NearestTree": { "X": 0.0, "Y": 0.0, "Z": 0.0 },
  "NearestOre": { "X": 0.0, "Y": 0.0, "Z": 0.0 }
}
```
*Note: Entities use relative vectors from the player.*