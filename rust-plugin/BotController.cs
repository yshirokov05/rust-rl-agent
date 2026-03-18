using System.IO;
using Newtonsoft.Json;
using UnityEngine;
using Rust;

namespace Carbon.Plugins
{
    [Info("BotController", "Antigravity", "0.1.0")]
    [Description("Spawns a server-side bot controlled by the RL agent via shared files.")]
    public class BotController : CarbonPlugin
    {
        private const string ActionsPath = "../../../../shared-data/actions.json";
        private const string VisionPath = "../../../../shared-data/vision.json";

        private BasePlayer _bot;
        private float _moveSpeed = 5f;

        private void OnServerInitialized()
        {
            Puts("BotController: Server initialized, spawning bot in 5 seconds...");
            timer.Once(5f, SpawnBot);
        }

        private void SpawnBot()
        {
            Vector3 spawnPos = ServerMgr.FindSpawnPoint().pos;
            _bot = GameManager.server.CreateEntity("assets/prefabs/player/player.prefab", spawnPos, Quaternion.identity) as BasePlayer;

            if (_bot == null)
            {
                Puts("BotController: ERROR - Failed to create bot entity!");
                return;
            }

            _bot.Spawn();
            _bot.displayName = "RL_Agent";
            _bot.health = 100f;
            _bot.metabolism.calories.value = 1000;
            _bot.metabolism.hydration.value = 1000;

            Puts($"BotController: Bot spawned at {spawnPos}");
            timer.Every(0.1f, ProcessActions);
        }

        private void ProcessActions()
        {
            if (_bot == null || _bot.IsDead()) 
            {
                SpawnBot();
                return;
            }

            _bot.health = 100f;
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

                float speedMult = actions.Sprint > 0.5f ? 1.7f : 1.0f;
                Vector3 moveDir = (forward * actions.Forward + right * actions.Strafe).normalized;
                Vector3 newPos = currentPos + moveDir * _moveSpeed * speedMult * 0.1f;
                newPos.y = TerrainMeta.HeightMap.GetHeight(newPos);
                _bot.MovePosition(newPos);

                if (Mathf.Abs(actions.Strafe) > 0.1f)
                {
                    _bot.transform.rotation *= Quaternion.Euler(0, actions.Strafe * 45f * 0.1f, 0);
                    _bot.SendNetworkUpdateImmediate();
                }

                if (actions.Attack > 0.5f) TryGatherNearest();
                WriteVisionData();
            }
            catch { }
        }

        private void TryGatherNearest()
        {
            if (_bot == null) return;
            float gatherRadius = 3f;
            var entities = Facepunch.Pool.Get<System.Collections.Generic.List<BaseEntity>>();
            Vis.Entities(_bot.transform.position, gatherRadius, entities);

            foreach (var entity in entities)
            {
                if (entity is TreeEntity || entity.ShortPrefabName.Contains("ore") || entity.ShortPrefabName.Contains("resource"))
                {
                    entity.OnAttacked(new HitInfo(_bot, entity, DamageType.Slash, 50f));
                    _hasGatheredThisTick = true;
                    break;
                }
            }
            Facepunch.Pool.FreeUnmanaged(ref entities);
        }

        private bool _hasGatheredThisTick = false;

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
                HasGathered = _hasGatheredThisTick
            };

            _hasGatheredThisTick = false;

            try
            {
                string json = JsonConvert.SerializeObject(visionData, Formatting.Indented);
                File.WriteAllText(VisionPath, json);
            }
            catch { }
        }

        private EntityInfo GetNearestEntity(Vector3 origin, string type)
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
                    if (dist < minDist) { minDist = dist; nearest = entity as BaseEntity; }
                }
            }

            if (nearest == null) return new EntityInfo(Vector3.zero, "None");
            return new EntityInfo(nearest.transform.position - origin, nearest.ShortPrefabName);
        }

        private void Unload() { if (_bot != null && !_bot.IsDestroyed) _bot.Kill(); }

        public class AgentActions { 
            public float Forward { get; set; } 
            public float Strafe { get; set; } 
            public float Jump { get; set; } 
            public float Attack { get; set; } 
            public float Sprint { get; set; }
        }
        public class VisionData { public Vec3 PlayerPosition { get; set; } public EntityInfo NearestTree { get; set; } public EntityInfo NearestOre { get; set; } public float Health { get; set; } public bool HasGathered { get; set; } }
        public class EntityInfo { public Vec3 Position { get; set; } public string Name { get; set; } public EntityInfo(Vector3 relativePos, string name) { Position = new Vec3(relativePos); Name = name; } }
        public class Vec3 { public float X { get; set; } public float Y { get; set; } public float Z { get; set; } public Vec3(float x, float y, float z) { X = x; Y = y; Z = z; } public Vec3(Vector3 v) { X = v.x; Y = v.y; Z = v.z; } }
    }
}
