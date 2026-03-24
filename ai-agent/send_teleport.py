import json
import os
import time

ACTIONS_PATH = r"C:\Projects\rust-rl-agent\shared-data\actions_0.json"
TARGET = {"X": 467.5, "Y": 6.9, "Z": 949.0}

print(f"🚀 [RCON MOCK] Injecting Teleport request for Bot_0 to {TARGET}...")

# Force write to actions_0.json
data = {
    "Movement": [0, 0],
    "Look": [0, 0],
    "Jump": False,
    "Attack": False,
    "Reload": False,
    "Inventory": False,
    "Respawn": True,
    "TeleportPos": TARGET
}

with open(ACTIONS_PATH, 'w') as f:
    json.dump(data, f)

print("✅ Injection complete. Carbon plugin should process this on next tick.")
