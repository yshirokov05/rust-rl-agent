using System.IO;
using Newtonsoft.Json;
using UnityEngine;

namespace Carbon.Plugins
{
    [Info("BotController", "Antigravity", "0.1.0")]
    [Description("Spawns a server-side bot controlled by the RL agent via shared files.")]
    public class BotController : CarbonPlugin
    {
        private const string ActionsPath = "../../../../../../shared-data/actions.json";
        private const string VisionPath = "../../../../../../shared-data/vision.json";

        private BasePlayer _bot;
        private float _moveSpeed = 5f;

        private void OnServerInitialized()
        {
            Puts("BotController: Server initialized, spawning bot in 5 seconds...");
            timer.Once(5f, SpawnBot);
        }

        private void SpawnBot()
        {
            // Find a valid spawn point on the map
            Vector3 spawnPos = ServerMgr.FindSpawnPoint().pos;
            
            // Create and spawn the bot
            _bot = GameManager.server.CreateEntity(
                "assets/prefabs/player/player.prefab",
                spawnPos,
                Quaternion.identity
            ) as BasePlayer;

            if (_bot == null)
            {
                Puts("BotController: ERROR - Failed to create bot entity!");
                return;
            }

            _bot.Spawn();
            _bot.displayName = "RL_Agent";

            // Give the bot god mode so it doesn't die during training
            _bot.metabolism.calories.max = 1000;
            _bot.metabolism.calories.value = 1000;
            _bot.metabolism.hydration.max = 1000;
            _bot.metabolism.hydration.value = 1000;
            _bot.health = 100f;

            Puts($"BotController: Bot spawned at {spawnPos}");

            // Start the action loop (10 ticks/sec matching AgentEyes)
            timer.Every(0.1f, ProcessActions);
        }

        private void ProcessActions()
        {
            if (_bot == null || _bot.IsDead()) 
            {
                // Respawn if dead
                SpawnBot();
                return;
            }

            // Keep the bot alive
            _bot.health = 100f;
            _bot.metabolism.calories.value = 1000;
            _bot.metabolism.hydration.value = 1000;

            // Read actions from the Python agent
            if (!File.Exists(ActionsPath)) return;

            try
            {
                string json = File.ReadAllText(ActionsPath);
                var actions = JsonConvert.DeserializeObject<AgentActions>(json);
                if (actions == null) return;

                // Apply movement
                Vector3 currentPos = _bot.transform.position;
                Vector3 forward = _bot.transform.forward;
                Vector3 right = _bot.transform.right;

                // actions.Forward: -1 (backward) to 1 (forward)
                // actions.Strafe: -1 (left) to 1 (right)
                Vector3 moveDir = (forward * actions.Forward + right * actions.Strafe).normalized;
                Vector3 newPos = currentPos + moveDir * _moveSpeed * 0.1f;

                // Keep the bot on the ground
                float terrainHeight = TerrainMeta.HeightMap.GetHeight(newPos);
                newPos.y = terrainHeight;

                _bot.MovePosition(newPos);

                // Handle rotation (turning left/right)
                if (Mathf.Abs(actions.Strafe) > 0.1f)
                {
                    Quaternion rotation = Quaternion.Euler(0, actions.Strafe * 45f * 0.1f, 0);
                    _bot.transform.rotation *= rotation;
                    _bot.SendNetworkUpdateImmediate();
                }

                // Handle attack/interact
                if (actions.Attack > 0.5f)
                {
                    // Try to gather the nearest resource
                    TryGatherNearest();
                }

                // Write vision data
                WriteVisionData();
            }
            catch (System.Exception ex)
            {
                // Silently handle file read conflicts
            }
        }

        private void TryGatherNearest()
        {
            if (_bot == null) return;

            // Find the nearest gatherable entity within 3 meters
            float gatherRadius = 3f;
            var entities = Facepunch.Pool.Get<System.Collections.Generic.List<BaseEntity>>();
            Vis.Entities(_bot.transform.position, gatherRadius, entities);

            foreach (var entity in entities)
            {
                if (entity is TreeEntity || entity.ShortPrefabName.Contains("ore"))
                {
                    // Simulate a gather hit
                    entity.OnAttacked(new HitInfo(_bot, entity, DamageType.Slash, 50f));
                    break;
                }
            }

            Facepunch.Pool.FreeUnmanaged(ref entities);
        }

        private void WriteVisionData()
        {
            if (_bot == null) return;

            var botPos = _bot.transform.position;

            var visionData = new VisionData
            {
                PlayerPosition = new Vec3(botPos),
                NearestTree = GetNearestEntity(botPos, "tree"),
                NearestOre = GetNearestEntity(botPos, "ore"),
                Health = _bot.health,
                HasGathered = false
            };

            try
            {
                string json = JsonConvert.SerializeObject(visionData, Formatting.Indented);
                File.WriteAllText(VisionPath, json);
            }
            catch { }
        }

        private Vec3 GetNearestEntity(Vector3 origin, string type)
        {
            BaseEntity nearest = null;
            float minDist = float.MaxValue;

            foreach (var entity in BaseEntity.serverEntities)
            {
                if (entity == null) continue;

                bool match = false;
                if (type == "tree" && (entity.ShortPrefabName.Contains("tree") || entity is TreeEntity)) match = true;
                if (type == "ore" && (entity.ShortPrefabName.Contains("ore") || entity.ShortPrefabName.Contains("resource"))) match = true;

                if (match)
                {
                    float dist = Vector3.Distance(origin, entity.transform.position);
                    if (dist < minDist)
                    {
                        minDist = dist;
                        nearest = entity;
                    }
                }
            }

            if (nearest == null) return new Vec3(0, 0, 0);

            Vector3 relative = nearest.transform.position - origin;
            return new Vec3(relative);
        }

        private void Unload()
        {
            if (_bot != null && !_bot.IsDestroyed)
            {
                _bot.Kill();
                Puts("BotController: Bot removed.");
            }
        }

        // Data classes
        public class AgentActions
        {
            public float Forward { get; set; }
            public float Strafe { get; set; }
            public float Jump { get; set; }
            public float Attack { get; set; }
        }

        public class VisionData
        {
            public Vec3 PlayerPosition { get; set; }
            public Vec3 NearestTree { get; set; }
            public Vec3 NearestOre { get; set; }
            public float Health { get; set; }
            public bool HasGathered { get; set; }
        }

        public class Vec3
        {
            public float X { get; set; }
            public float Y { get; set; }
            public float Z { get; set; }

            public Vec3(float x, float y, float z) { X = x; Y = y; Z = z; }
            public Vec3(Vector3 v) { X = v.x; Y = v.y; Z = v.z; }
        }
    }
}
