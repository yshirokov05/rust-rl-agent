using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using Newtonsoft.Json;
using UnityEngine;
using Rust;

namespace Carbon.Plugins
{
    [Info("BotController", "RustRL", "1.1.0")]
    [Description("Private-server MVP controller and telemetry bridge for Rust RL training.")]
    public class BotController : CarbonPlugin
    {
        private const int ProtocolVersion = 1;
        private const float ActionInterval = 0.1f;
        private const ulong BotUserIdBase = 70000000000000000UL;

        private sealed class BotRuntimeState
        {
            public readonly int Index;
            public readonly Vector3 SpawnPoint;

            public BasePlayer Bot;
            public string SessionId = string.Empty;
            public int AppliedStepId = -1;
            public int LastAcceptedTick = -1;
            public bool InventoryInitialized;
            public int LastWood;
            public int LastStone;
            public int PendingWoodDelta;
            public int PendingStoneDelta;
            public string LastError = string.Empty;

            public BotRuntimeState(int index, Vector3 spawnPoint)
            {
                Index = index;
                SpawnPoint = spawnPoint;
            }
        }

        private readonly List<BotRuntimeState> _botStates = new List<BotRuntimeState>();
        private readonly List<BaseEntity> _resourceCache = new List<BaseEntity>();
        private readonly Dictionary<string, float> _nextWarningTime = new Dictionary<string, float>();

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

            try
            {
                Directory.CreateDirectory(_sharedDataPath);
            }
            catch (Exception exception)
            {
                PrintError(
                    "RustRL: cannot create shared-data directory " +
                    _sharedDataPath + ": " + exception.Message
                );
                return;
            }

            CleanupLegacyBots();
            for (int index = 0; index < _botCount; index++)
            {
                var state = new BotRuntimeState(index, GetSpawnPoint(index));
                _botStates.Add(state);
                SpawnCanonicalBot(state);
            }

            timer.Every(ActionInterval, ProcessActionsAndPublish);
            Puts(
                "RustRL: initialized " + _botCount +
                " bot(s), shared data at " + _sharedDataPath +
                ", invulnerable=" + _invulnerable +
                ". A new SessionId with Reset=true and StepId=0 is required."
            );
        }

        private Vector3 GetSpawnPoint(int index)
        {
            float spawnX = 195.0f + (index * 1.5f);
            float spawnZ = 145.0f;
            Vector3 spawnPoint = new Vector3(spawnX, 0f, spawnZ);
            spawnPoint.y = TerrainMeta.HeightMap.GetHeight(spawnPoint);
            return spawnPoint;
        }

        private void CleanupLegacyBots()
        {
            var toKill = new List<BaseEntity>();
            foreach (var player in BasePlayer.activePlayerList)
            {
                if (IsOwnedBot(player))
                {
                    toKill.Add(player);
                }
            }
            foreach (var player in BasePlayer.sleepingPlayerList)
            {
                if (IsOwnedBot(player) && !toKill.Contains(player))
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

        private static bool IsOwnedBot(BasePlayer player)
        {
            return player != null &&
                   player.displayName != null &&
                   player.displayName.StartsWith(
                       "RL_Agent_",
                       StringComparison.Ordinal
                   );
        }

        private bool SpawnCanonicalBot(BotRuntimeState state)
        {
            var bot = GameManager.server.CreateEntity(
                "assets/prefabs/player/player.prefab",
                state.SpawnPoint,
                Quaternion.identity
            ) as BasePlayer;

            if (bot == null)
            {
                SetStateError(
                    state,
                    "spawn",
                    "failed to create player prefab for bot " + state.Index
                );
                return false;
            }

            // Set identity before Spawn so the server never sees duplicate
            // userID zero bots and diagnostics have a stable display name.
            bot.userID = BotUserIdBase + (ulong)state.Index;
            bot.displayName = "RL_Agent_" + state.Index;
            bot.syncPosition = true;
            bot.Spawn();

            var movement = bot.GetComponent<PlayerWalkMovement>();
            if (movement != null)
            {
                UnityEngine.Object.Destroy(movement);
            }

            state.Bot = bot;
            if (!RestoreCanonicalState(state))
            {
                return false;
            }

            Puts(
                "RustRL: spawned RL_Agent_" + state.Index +
                " at " + state.SpawnPoint
            );
            return true;
        }

        private bool ResetBot(BotRuntimeState state)
        {
            var bot = state.Bot;
            if (bot == null || bot.IsDestroyed || bot.IsDead())
            {
                if (bot != null && !bot.IsDestroyed)
                {
                    bot.Kill();
                }
                state.Bot = null;
                return SpawnCanonicalBot(state);
            }

            return RestoreCanonicalState(state);
        }

        private bool RestoreCanonicalState(BotRuntimeState state)
        {
            var bot = state.Bot;
            if (bot == null || bot.IsDestroyed)
            {
                SetStateError(
                    state,
                    "reset-missing-bot",
                    "cannot reset missing bot " + state.Index
                );
                return false;
            }

            try
            {
                bot.EndSleeping();
                bot.MovePosition(state.SpawnPoint);
                bot.transform.rotation = Quaternion.identity;
                bot.viewAngles = Vector3.zero;
                bot.TransformChanged();

                float canonicalHealth = _invulnerable ? 99999f : 100f;
                bot.InitializeHealth(canonicalHealth, canonicalHealth);
                bot.health = canonicalHealth;
                RestoreMetabolism(bot);

                bot.modelState.flags |= (int)ModelState.Flag.OnGround;
                bot.modelState.flags &= ~(int)ModelState.Flag.Sprinting;
                bot.modelState.flags &= ~(int)ModelState.Flag.Jumped;

                if (bot.inventory == null || bot.inventory.containerBelt == null)
                {
                    SetStateError(
                        state,
                        "reset-inventory",
                        "inventory was unavailable for bot " + state.Index
                    );
                    return false;
                }

                bot.inventory.Strip();
                var rock = ItemManager.CreateByName("rock", 1);
                if (rock == null)
                {
                    SetStateError(
                        state,
                        "reset-rock",
                        "failed to create starter rock for bot " + state.Index
                    );
                    return false;
                }

                rock.MoveToContainer(bot.inventory.containerBelt, 0);
                bot.UpdateActiveItem(rock.uid);
                bot.SendNetworkUpdateImmediate();

                var activeItem = bot.GetActiveItem();
                if (activeItem == null ||
                    activeItem.info == null ||
                    activeItem.info.shortname != "rock")
                {
                    SetStateError(
                        state,
                        "reset-equip",
                        "starter rock was not active for bot " + state.Index
                    );
                    return false;
                }

                state.InventoryInitialized = true;
                state.LastWood = GetInventoryAmount(bot, "wood");
                state.LastStone = GetInventoryAmount(bot, "stones");
                state.PendingWoodDelta = 0;
                state.PendingStoneDelta = 0;
                state.LastError = string.Empty;
                return true;
            }
            catch (Exception exception)
            {
                SetStateError(
                    state,
                    "reset-exception",
                    "canonical reset failed for bot " + state.Index +
                    ": " + exception.Message
                );
                return false;
            }
        }

        private void ProcessActionsAndPublish()
        {
            _tickCount++;

            if (Time.time >= _nextResourceScan)
            {
                try
                {
                    RefreshResourceCache();
                }
                catch (Exception exception)
                {
                    WarnRateLimited(
                        "resource-scan",
                        "RustRL: resource scan failed: " + exception.Message
                    );
                }
                _nextResourceScan = Time.time + 1.0f;
            }

            foreach (var state in _botStates)
            {
                try
                {
                    ProcessBot(state);
                }
                catch (Exception exception)
                {
                    SetStateError(
                        state,
                        "tick-exception",
                        "bot " + state.Index + " tick failed: " +
                        exception.Message
                    );
                    PublishTelemetrySafe(state);
                }
            }
        }

        private void ProcessBot(BotRuntimeState state)
        {
            var bot = state.Bot;
            if (IsAlive(bot))
            {
                CaptureInventoryDeltas(state, bot);
                if (_invulnerable)
                {
                    bot.health = 99999f;
                    RestoreMetabolism(bot);
                }
            }

            Dictionary<string, object> actions;
            if (TryReadAction(state, out actions))
            {
                TryAcceptAction(state, actions);
            }

            bot = state.Bot;
            if (IsAlive(bot))
            {
                // Capture both synchronous changes and delayed strikes that
                // completed after a previous action tick.
                CaptureInventoryDeltas(state, bot);
            }

            PublishTelemetrySafe(state);
        }

        private bool TryReadAction(
            BotRuntimeState state,
            out Dictionary<string, object> actions
        )
        {
            actions = null;
            string path = Path.Combine(
                _sharedDataPath,
                "actions_" + state.Index + ".json"
            );

            if (!File.Exists(path))
            {
                return false;
            }

            try
            {
                actions = JsonConvert.DeserializeObject<Dictionary<string, object>>(
                    File.ReadAllText(path)
                );
                if (actions == null)
                {
                    SetStateError(
                        state,
                        "action-null",
                        "action file contained JSON null for bot " + state.Index
                    );
                    return false;
                }
                return true;
            }
            catch (Exception exception)
            {
                SetStateError(
                    state,
                    "action-read",
                    "failed to read action file for bot " + state.Index +
                    ": " + exception.Message
                );
                return false;
            }
        }

        private void TryAcceptAction(
            BotRuntimeState state,
            Dictionary<string, object> actions
        )
        {
            int actionProtocol = ReadInt(actions, "ProtocolVersion", -1);
            int actionBotId = ReadInt(actions, "BotId", -1);
            int stepId = ReadInt(actions, "StepId", -1);
            string sessionId = ReadString(actions, "SessionId", string.Empty);
            bool reset = ReadBool(actions, "Reset", false);

            if (actionProtocol != ProtocolVersion)
            {
                RejectAction(
                    state,
                    "protocol",
                    "expected ProtocolVersion " + ProtocolVersion +
                    ", received " + actionProtocol
                );
                return;
            }
            if (actionBotId != state.Index)
            {
                RejectAction(
                    state,
                    "bot-id",
                    "expected BotId " + state.Index +
                    ", received " + actionBotId
                );
                return;
            }
            if (string.IsNullOrWhiteSpace(sessionId))
            {
                RejectAction(state, "session", "SessionId must be non-empty");
                return;
            }
            if (stepId < 0)
            {
                RejectAction(state, "step-negative", "StepId must be non-negative");
                return;
            }

            if (reset)
            {
                if (stepId != 0)
                {
                    RejectAction(
                        state,
                        "reset-step",
                        "Reset=true requires StepId=0"
                    );
                    return;
                }

                if (sessionId == state.SessionId)
                {
                    // A duplicate reset file is already acknowledged. Do not
                    // clear inventory repeatedly while Python is initializing.
                    return;
                }

                if (!ResetBot(state))
                {
                    return;
                }

                state.SessionId = sessionId;
                state.AppliedStepId = 0;
                state.LastAcceptedTick = _tickCount;
                state.LastError = string.Empty;
                Puts(
                    "RustRL: bot " + state.Index +
                    " reset for session " + sessionId
                );
                return;
            }

            if (sessionId != state.SessionId)
            {
                RejectAction(
                    state,
                    "session-change",
                    "a new SessionId must begin with Reset=true and StepId=0"
                );
                return;
            }

            if (stepId <= state.AppliedStepId)
            {
                // Re-reading an accepted atomic file is expected while PPO
                // updates. Never apply that physical action twice.
                return;
            }

            int expectedStep = state.AppliedStepId + 1;
            if (stepId != expectedStep)
            {
                RejectAction(
                    state,
                    "step-gap",
                    "expected StepId " + expectedStep +
                    ", received " + stepId
                );
                return;
            }

            if (!IsAlive(state.Bot))
            {
                RejectAction(
                    state,
                    "bot-dead",
                    "bot is unavailable; start a new session with Reset=true"
                );
                return;
            }

            // Accept before applying. If a version-sensitive Rust API throws
            // after partially moving the bot, the physical action must not run
            // again on the next 100 ms tick.
            state.AppliedStepId = stepId;
            state.LastAcceptedTick = _tickCount;
            state.LastError = string.Empty;
            try
            {
                ApplyAction(state, actions);
            }
            catch (Exception exception)
            {
                SetStateError(
                    state,
                    "action-apply",
                    "accepted StepId " + stepId +
                    " but action application failed: " + exception.Message
                );
            }
        }

        private void RejectAction(
            BotRuntimeState state,
            string category,
            string reason
        )
        {
            SetStateError(
                state,
                "reject-" + category,
                "rejected action for bot " + state.Index + ": " + reason
            );
        }

        private void ApplyAction(
            BotRuntimeState state,
            Dictionary<string, object> actions
        )
        {
            var bot = state.Bot;
            float moveX = ReadBoundedFloat(actions, "MoveX", -1f, 1f, 0f);
            float moveZ = ReadBoundedFloat(actions, "MoveZ", -1f, 1f, 0f);
            float lookX = ReadBoundedFloat(actions, "LookX", -1f, 1f, 0f);
            float lookY = ReadBoundedFloat(actions, "LookY", -1f, 1f, 0f);
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

            Vector3 localInput = Vector3.ClampMagnitude(
                new Vector3(moveX, 0f, moveZ),
                1f
            );
            if (localInput.sqrMagnitude > 0.0025f)
            {
                Vector3 direction =
                    bot.transform.forward * localInput.z +
                    bot.transform.right * localInput.x;
                float speed = 5f * (sprint ? 1.4f : 1f);
                Vector3 nextPosition =
                    bot.transform.position + direction * speed * ActionInterval;
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
                var activeItem = bot.GetActiveItem();
                var melee = activeItem == null
                    ? null
                    : activeItem.GetHeldEntity() as BaseMelee;
                if (melee == null)
                {
                    SetStateError(
                        state,
                        "attack-held-entity",
                        "Attack requested but the active item had no BaseMelee held entity"
                    );
                }
                else
                {
                    // ServerUse is Rust's server-side AI melee path. Its exact
                    // gather behavior must be proven on the installed build.
                    melee.ServerUse();
                }
            }

            bot.SendNetworkUpdateImmediate();
        }

        private void CaptureInventoryDeltas(
            BotRuntimeState state,
            BasePlayer bot
        )
        {
            int wood = GetInventoryAmount(bot, "wood");
            int stone = GetInventoryAmount(bot, "stones");

            if (!state.InventoryInitialized)
            {
                state.LastWood = wood;
                state.LastStone = stone;
                state.InventoryInitialized = true;
                return;
            }

            int woodDelta = wood - state.LastWood;
            int stoneDelta = stone - state.LastStone;
            if (woodDelta > 0)
            {
                state.PendingWoodDelta = SaturatingAdd(
                    state.PendingWoodDelta,
                    woodDelta
                );
            }
            if (stoneDelta > 0)
            {
                state.PendingStoneDelta = SaturatingAdd(
                    state.PendingStoneDelta,
                    stoneDelta
                );
            }

            state.LastWood = wood;
            state.LastStone = stone;
        }

        private static int SaturatingAdd(int left, int right)
        {
            if (right > 0 && left > int.MaxValue - right)
            {
                return int.MaxValue;
            }
            return left + right;
        }

        private void WriteTelemetry(BotRuntimeState state)
        {
            var bot = state.Bot;
            bool alive = IsAlive(bot);
            int woodDelta = state.PendingWoodDelta;
            int stoneDelta = state.PendingStoneDelta;

            var payload = new Dictionary<string, object>
            {
                { "ProtocolVersion", ProtocolVersion },
                { "BotId", state.Index },
                { "Tick", _tickCount },
                { "AppliedStepId", state.AppliedStepId },
                { "SessionId", state.SessionId },
                { "LastAcceptedTick", state.LastAcceptedTick },
                { "ResetRequired", string.IsNullOrEmpty(state.SessionId) || !alive },
                { "Alive", alive },
                { "HasGathered", woodDelta > 0 || stoneDelta > 0 },
                { "WoodDelta", woodDelta },
                { "StoneDelta", stoneDelta },
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
                { "SemanticMapBase64", string.Empty },
                { "LastError", state.LastError }
            };

            if (WriteJsonAtomic(
                state,
                Path.Combine(
                    _sharedDataPath,
                    "vision_" + state.Index + ".json"
                ),
                payload
            ))
            {
                // Keep deltas across a failed replacement and clear them only
                // after one complete telemetry payload is published.
                state.PendingWoodDelta = 0;
                state.PendingStoneDelta = 0;
            }
        }

        private void PublishTelemetrySafe(BotRuntimeState state)
        {
            try
            {
                WriteTelemetry(state);
            }
            catch (Exception exception)
            {
                SetStateError(
                    state,
                    "telemetry-build",
                    "failed to build telemetry for bot " + state.Index +
                    ": " + exception.Message
                );
            }
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

            Vector3 worldRelative =
                nearest.transform.position - bot.transform.position;
            Vector3 botLocal =
                bot.transform.InverseTransformDirection(worldRelative);
            return new Dictionary<string, object>
            {
                { "Name", GetPrefabName(nearest) },
                { "Distance", nearestDistance },
                { "Position", Position(botLocal) }
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
                   name.Contains("cactus") ||
                   name.Contains("oak") ||
                   name.Contains("douglas");
        }

        private static bool IsOreName(string name)
        {
            return name.Contains("stone-ore") ||
                   name.Contains("metal-ore") ||
                   name.Contains("sulfur-ore") ||
                   name.Contains("ore.prefab");
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

        private static bool IsAlive(BasePlayer bot)
        {
            return bot != null && !bot.IsDestroyed && !bot.IsDead();
        }

        private static void RestoreMetabolism(BasePlayer bot)
        {
            if (bot.metabolism == null)
            {
                return;
            }
            bot.metabolism.calories.value = 1000f;
            bot.metabolism.hydration.value = 1000f;
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
                float value = Convert.ToSingle(raw, CultureInfo.InvariantCulture);
                return float.IsNaN(value) || float.IsInfinity(value)
                    ? fallback
                    : value;
            }
            catch (Exception)
            {
                return fallback;
            }
        }

        private static float ReadBoundedFloat(
            Dictionary<string, object> values,
            string key,
            float minimum,
            float maximum,
            float fallback
        )
        {
            return Mathf.Clamp(
                ReadFloat(values, key, fallback),
                minimum,
                maximum
            );
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

        private bool WriteJsonAtomic(
            BotRuntimeState state,
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
                return true;
            }
            catch (Exception exception)
            {
                SetStateError(
                    state,
                    "telemetry-write",
                    "failed to publish telemetry for bot " + state.Index +
                    " to " + path + ": " + exception.Message
                );
                try
                {
                    if (File.Exists(temporary))
                    {
                        File.Delete(temporary);
                    }
                }
                catch (Exception cleanupException)
                {
                    WarnRateLimited(
                        "telemetry-cleanup-" + state.Index,
                        "RustRL: failed to remove temporary telemetry file " +
                        temporary + ": " + cleanupException.Message
                    );
                }
                return false;
            }
        }

        private void SetStateError(
            BotRuntimeState state,
            string category,
            string message
        )
        {
            state.LastError = message;
            WarnRateLimited(
                category + "-" + state.Index,
                "RustRL: " + message
            );
        }

        private void WarnRateLimited(
            string key,
            string message,
            float intervalSeconds = 5f
        )
        {
            float now = Time.realtimeSinceStartup;
            float next;
            if (_nextWarningTime.TryGetValue(key, out next) && now < next)
            {
                return;
            }

            _nextWarningTime[key] = now + intervalSeconds;
            PrintWarning(message);
        }

        private void Unload()
        {
            foreach (var state in _botStates)
            {
                var bot = state.Bot;
                if (bot != null && !bot.IsDestroyed)
                {
                    bot.Kill();
                }
            }
            _botStates.Clear();
        }
    }
}
