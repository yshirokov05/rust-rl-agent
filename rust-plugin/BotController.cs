using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using Newtonsoft.Json;
using UnityEngine;
using Rust;

namespace Carbon.Plugins
{
    [Info("BotController", "RustRL", "1.0.0")]
    [Description("Private-server MVP controller and telemetry bridge for Rust RL training.")]
    public class BotController : CarbonPlugin
    {
        private const int ProtocolVersion = 1;
        private const float ActionInterval = 0.1f;

        private readonly List<BasePlayer> _bots = new List<BasePlayer>();
        private readonly Dictionary<int, int> _lastWood = new Dictionary<int, int>();
        private readonly Dictionary<int, int> _lastStone = new Dictionary<int, int>();
        private readonly List<BaseEntity> _resourceCache = new List<BaseEntity>();

        private string _sharedDataPath;
        private int _botCount = 1;
        private bool _invulnerable = true;
        private int _tickCount;
        private float _nextResourceScan;

        private void OnServerInitialized()
        {
            _sharedDataPath = Environment.GetEnvironmentVariable("RUST_RL_SHARED_DATA");
            if (string.IsNullOrWhiteSpace(_sharedDataPath))
            {
                _sharedDataPath = "C:/Projects/rust-rl-agent/shared-data";
            }

            _botCount = ReadEnvironmentInt("RUST_RL_BOT_COUNT", 1);
            _botCount = Mathf.Clamp(_botCount, 1, 16);
            _invulnerable = ReadEnvironmentBool("RUST_RL_INVULNERABLE", true);
            Directory.CreateDirectory(_sharedDataPath);

            CleanupLegacyBots();
            for (int index = 0; index < _botCount; index++)
            {
                SpawnBot(index);
            }

            timer.Every(ActionInterval, ProcessActionsAndPublish);
            Puts(
                "RustRL: initialized " + _botCount +
                " bot(s), shared data at " + _sharedDataPath +
                ", invulnerable=" + _invulnerable
            );
        }

        private void CleanupLegacyBots()
        {
            var toKill = new List<BaseEntity>();
            foreach (var player in BasePlayer.activePlayerList)
            {
                if (player != null && player.displayName.Contains("RL_Agent"))
                {
                    toKill.Add(player);
                }
            }
            foreach (var player in BasePlayer.sleepingPlayerList)
            {
                if (player != null && player.displayName.Contains("RL_Agent"))
                {
                    toKill.Add(player);
                }
            }
            foreach (var entity in toKill)
            {
                if (entity != null && !entity.IsDestroyed)
                {
                    entity.Kill();
                }
            }
        }

        private void SpawnBot(int index)
        {
            float spawnX = 195.0f + (index * 1.5f);
            float spawnZ = 145.0f;
            Vector3 spawnPoint = new Vector3(spawnX, 0, spawnZ);
            spawnPoint.y = TerrainMeta.HeightMap.GetHeight(spawnPoint);

            var bot = GameManager.server.CreateEntity(
                "assets/prefabs/player/player.prefab",
                spawnPoint,
                Quaternion.identity
            ) as BasePlayer;

            if (bot == null)
            {
                Puts("RustRL: failed to create bot " + index);
                return;
            }

            bot.Spawn();
            var movement = bot.GetComponent<PlayerWalkMovement>();
            if (movement != null)
            {
                UnityEngine.Object.Destroy(movement);
            }

            bot.EndSleeping();
            bot.displayName = "RL_Agent_" + index;
            bot.InitializeHealth(_invulnerable ? 99999f : 100f, _invulnerable ? 99999f : 100f);
            bot.health = _invulnerable ? 99999f : 100f;
            bot.metabolism.calories.value = 1000f;
            bot.metabolism.hydration.value = 1000f;

            timer.Once(1f, delegate
            {
                if (bot == null || bot.IsDestroyed)
                {
                    return;
                }

                bot.EndSleeping();
                var rock = ItemManager.CreateByName("rock", 1);
                if (rock != null)
                {
                    rock.MoveToContainer(bot.inventory.containerBelt, 0);
                    bot.UpdateActiveItem(rock.uid);
                    bot.SendNetworkUpdateImmediate();
                }
            });

            _bots.Add(bot);
            Puts("RustRL: spawned RL_Agent_" + index + " at " + spawnPoint);
        }

        private void ProcessActionsAndPublish()
        {
            _tickCount++;
            if (Time.time >= _nextResourceScan)
            {
                RefreshResourceCache();
                _nextResourceScan = Time.time + 1.0f;
            }

            for (int index = 0; index < _bots.Count; index++)
            {
                var bot = _bots[index];
                var actions = ReadAction(index);

                if (bot == null || bot.IsDestroyed)
                {
                    WriteTelemetry(index, null, actions, false);
                    continue;
                }

                if (bot.IsDead())
                {
                    WriteTelemetry(index, bot, actions, false);
                    continue;
                }

                int woodBefore = GetInventoryAmount(bot, "wood");
                int stoneBefore = GetInventoryAmount(bot, "stones");

                if (_invulnerable)
                {
                    bot.health = 99999f;
                    bot.metabolism.calories.value = 1000f;
                    bot.metabolism.hydration.value = 1000f;
                }

                ApplyAction(bot, actions);

                int woodAfter = GetInventoryAmount(bot, "wood");
                int stoneAfter = GetInventoryAmount(bot, "stones");
                bool gathered = woodAfter > woodBefore || stoneAfter > stoneBefore;
                _lastWood[index] = woodAfter;
                _lastStone[index] = stoneAfter;

                WriteTelemetry(index, bot, actions, gathered);
            }
        }

        private Dictionary<string, object> ReadAction(int index)
        {
            string path = Path.Combine(_sharedDataPath, "actions_" + index + ".json");
            if (!File.Exists(path))
            {
                return new Dictionary<string, object>();
            }

            try
            {
                var parsed = JsonConvert.DeserializeObject<Dictionary<string, object>>(
                    File.ReadAllText(path)
                );
                return parsed ?? new Dictionary<string, object>();
            }
            catch (Exception)
            {
                return new Dictionary<string, object>();
            }
        }

        private void ApplyAction(
            BasePlayer bot,
            Dictionary<string, object> actions
        )
        {
            float moveX = ReadFloat(actions, "MoveX", ReadFloat(actions, "Strafe", 0f));
            float moveZ = ReadFloat(actions, "MoveZ", ReadFloat(actions, "Forward", 0f));
            float lookX = ReadFloat(actions, "LookX", 0f);
            float lookY = ReadFloat(actions, "LookY", 0f);
            bool sprint = ReadBool(actions, "Sprint", false);
            bool jump = ReadBool(actions, "Jump", false);
            bool attack = ReadBool(actions, "Attack", false);

            float yaw = bot.transform.eulerAngles.y + lookX * 8f;
            float pitch = NormalizeAngle(bot.viewAngles.x) - lookY * 5f;
            pitch = Mathf.Clamp(pitch, -80f, 80f);

            bot.viewAngles = new Vector3(pitch, yaw, 0f);
            bot.transform.rotation = Quaternion.Euler(0f, yaw, 0f);

            bot.modelState.flags |= (int)ModelState.Flag.OnGround;
            bot.modelState.flags &= ~(int)ModelState.Flag.Sprinting;
            bot.modelState.flags &= ~(int)ModelState.Flag.Jumped;

            if (Mathf.Abs(moveX) > 0.05f || Mathf.Abs(moveZ) > 0.05f)
            {
                Vector3 direction = (
                    bot.transform.forward * moveZ +
                    bot.transform.right * moveX
                ).normalized;
                float speed = 5f * (sprint ? 1.4f : 1f);
                Vector3 nextPosition = bot.transform.position + direction * speed * ActionInterval;
                nextPosition.y = TerrainMeta.HeightMap.GetHeight(nextPosition);
                bot.MovePosition(nextPosition);
                bot.TransformChanged();

                if (sprint)
                {
                    bot.modelState.flags |= (int)ModelState.Flag.Sprinting;
                }
            }

            if (jump)
            {
                bot.MovePosition(bot.transform.position + Vector3.up * 1.5f);
                bot.modelState.flags |= (int)ModelState.Flag.Jumped;
            }

            if (attack)
            {
                var melee = bot.GetActiveItem()?.GetHeldEntity() as BaseMelee;
                if (melee != null)
                {
                    melee.ServerUse();
                }
                bot.SignalBroadcast(BaseEntity.Signal.Attack, string.Empty);
            }

            bot.SendNetworkUpdateImmediate();
        }

        private void WriteTelemetry(
            int index,
            BasePlayer bot,
            Dictionary<string, object> actions,
            bool gathered
        )
        {
            bool alive = bot != null && !bot.IsDestroyed && !bot.IsDead();
            int appliedStep = ReadInt(actions, "StepId", -1);
            string sessionId = ReadString(actions, "SessionId", string.Empty);

            var payload = new Dictionary<string, object>
            {
                { "ProtocolVersion", ProtocolVersion },
                { "BotId", index },
                { "Tick", _tickCount },
                { "AppliedStepId", appliedStep },
                { "SessionId", sessionId },
                { "Alive", alive },
                { "HasGathered", gathered },
                { "Health", alive ? Mathf.Clamp(bot.health, 0f, 100f) : 0f },
                { "WoodCount", alive ? GetInventoryAmount(bot, "wood") : 0 },
                { "StoneCount", alive ? GetInventoryAmount(bot, "stones") : 0 },
                { "IsPredatorNearby", false },
                { "ActiveItem", alive ? GetActiveItemName(bot) : "none" },
                { "PlayerPosition", alive ? Position(bot.transform.position) : Position(Vector3.zero) },
                { "PlayerYaw", alive ? bot.transform.eulerAngles.y : 0f },
                { "PlayerPitch", alive ? bot.viewAngles.x : 0f },
                { "NearestTree", alive ? FindNearestResource(bot, true) : null },
                { "NearestOre", alive ? FindNearestResource(bot, false) : null },
                { "SemanticMapBase64", string.Empty }
            };

            WriteJsonAtomic(
                Path.Combine(_sharedDataPath, "vision_" + index + ".json"),
                payload
            );
        }

        private void RefreshResourceCache()
        {
            _resourceCache.Clear();
            foreach (var networkable in BaseNetworkable.serverEntities)
            {
                var entity = networkable as BaseEntity;
                if (entity == null || entity.IsDestroyed)
                {
                    continue;
                }

                string name = GetPrefabName(entity);
                if (IsTreeName(name) || IsOreName(name))
                {
                    _resourceCache.Add(entity);
                }
            }
        }

        private Dictionary<string, object> FindNearestResource(
            BasePlayer bot,
            bool tree
        )
        {
            BaseEntity nearest = null;
            float nearestDistance = float.MaxValue;

            foreach (var entity in _resourceCache)
            {
                if (entity == null || entity.IsDestroyed)
                {
                    continue;
                }

                string name = GetPrefabName(entity);
                bool matches = tree ? IsTreeName(name) : IsOreName(name);
                if (!matches)
                {
                    continue;
                }

                float distance = Vector3.Distance(
                    bot.transform.position,
                    entity.transform.position
                );
                if (distance < nearestDistance)
                {
                    nearest = entity;
                    nearestDistance = distance;
                }
            }

            if (nearest == null)
            {
                return null;
            }

            Vector3 relative = nearest.transform.position - bot.transform.position;
            return new Dictionary<string, object>
            {
                { "Name", GetPrefabName(nearest) },
                { "Distance", nearestDistance },
                { "Position", Position(relative) }
            };
        }

        private static string GetPrefabName(BaseEntity entity)
        {
            return (
                entity.ShortPrefabName ??
                entity.name ??
                string.Empty
            ).ToLowerInvariant();
        }

        private static bool IsTreeName(string name)
        {
            return name.Contains("tree") ||
                   name.Contains("pine") ||
                   name.Contains("birch") ||
                   name.Contains("palm") ||
                   name.Contains("cactus");
        }

        private static bool IsOreName(string name)
        {
            return name.Contains("ore") ||
                   name.Contains("stone-ore") ||
                   name.Contains("metal-ore") ||
                   name.Contains("sulfur-ore");
        }

        private static Dictionary<string, object> Position(Vector3 value)
        {
            return new Dictionary<string, object>
            {
                { "X", value.x },
                { "Y", value.y },
                { "Z", value.z }
            };
        }

        private static string GetActiveItemName(BasePlayer bot)
        {
            var item = bot.GetActiveItem();
            return item == null || item.info == null
                ? "none"
                : item.info.shortname ?? "none";
        }

        private static int GetInventoryAmount(BasePlayer bot, string shortName)
        {
            if (bot == null || bot.inventory == null)
            {
                return 0;
            }

            var definition = ItemManager.FindItemDefinition(shortName);
            return definition == null
                ? 0
                : bot.inventory.GetAmount(definition.itemid);
        }

        private static float ReadFloat(
            Dictionary<string, object> values,
            string key,
            float fallback
        )
        {
            object raw;
            if (!values.TryGetValue(key, out raw) || raw == null)
            {
                return fallback;
            }

            try
            {
                return Convert.ToSingle(raw, CultureInfo.InvariantCulture);
            }
            catch (Exception)
            {
                return fallback;
            }
        }

        private static int ReadInt(
            Dictionary<string, object> values,
            string key,
            int fallback
        )
        {
            object raw;
            if (!values.TryGetValue(key, out raw) || raw == null)
            {
                return fallback;
            }

            int result;
            return int.TryParse(
                Convert.ToString(raw, CultureInfo.InvariantCulture),
                NumberStyles.Integer,
                CultureInfo.InvariantCulture,
                out result
            ) ? result : fallback;
        }

        private static bool ReadBool(
            Dictionary<string, object> values,
            string key,
            bool fallback
        )
        {
            object raw;
            if (!values.TryGetValue(key, out raw) || raw == null)
            {
                return fallback;
            }

            if (raw is bool)
            {
                return (bool)raw;
            }

            bool parsed;
            if (bool.TryParse(Convert.ToString(raw), out parsed))
            {
                return parsed;
            }

            return ReadFloat(values, key, fallback ? 1f : 0f) > 0f;
        }

        private static string ReadString(
            Dictionary<string, object> values,
            string key,
            string fallback
        )
        {
            object raw;
            if (!values.TryGetValue(key, out raw) || raw == null)
            {
                return fallback;
            }
            return Convert.ToString(raw, CultureInfo.InvariantCulture) ?? fallback;
        }

        private static int ReadEnvironmentInt(string key, int fallback)
        {
            int value;
            return int.TryParse(
                Environment.GetEnvironmentVariable(key),
                NumberStyles.Integer,
                CultureInfo.InvariantCulture,
                out value
            ) ? value : fallback;
        }

        private static bool ReadEnvironmentBool(string key, bool fallback)
        {
            string raw = Environment.GetEnvironmentVariable(key);
            if (string.IsNullOrWhiteSpace(raw))
            {
                return fallback;
            }

            bool value;
            return bool.TryParse(raw, out value) ? value : fallback;
        }

        private static float NormalizeAngle(float angle)
        {
            while (angle > 180f)
            {
                angle -= 360f;
            }
            while (angle < -180f)
            {
                angle += 360f;
            }
            return angle;
        }

        private static void WriteJsonAtomic(
            string path,
            Dictionary<string, object> payload
        )
        {
            string temporary = path + "." + Guid.NewGuid().ToString("N") + ".tmp";
            try
            {
                File.WriteAllText(
                    temporary,
                    JsonConvert.SerializeObject(payload)
                );

                if (File.Exists(path))
                {
                    File.Replace(temporary, path, null);
                }
                else
                {
                    File.Move(temporary, path);
                }
            }
            catch (Exception)
            {
                try
                {
                    if (File.Exists(temporary))
                    {
                        File.Delete(temporary);
                    }
                }
                catch (Exception)
                {
                    // Best-effort cleanup; the next server tick retries.
                }
            }
        }

        private void Unload()
        {
            foreach (var bot in _bots)
            {
                if (bot != null && !bot.IsDestroyed)
                {
                    bot.Kill();
                }
            }
            _bots.Clear();
        }
    }
}
