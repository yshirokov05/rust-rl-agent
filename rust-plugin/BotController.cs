using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using Newtonsoft.Json;
using UnityEngine;
using Rust;

namespace Carbon.Plugins
{
    [Info("BotController", "Antigravity", "0.3.3")]
    [Description("Bitwise-only modelState puppeteer (v0.3.3).")]
    public class BotController : CarbonPlugin
    {
        private List<BasePlayer> _bots = new List<BasePlayer>();
        private int _tickCount = 0;

        private void OnServerInitialized()
        {
            // The Janitor: Cleanup legacy agents before spawning fresh ones
            var toKill = new List<BaseEntity>();
            foreach (var p in BasePlayer.activePlayerList) if (p != null && p.displayName.Contains("RL_Agent")) toKill.Add(p);
            foreach (var p in BasePlayer.sleepingPlayerList) if (p != null && p.displayName.Contains("RL_Agent")) toKill.Add(p);
            foreach (var e in toKill) if (e != null && !e.IsDestroyed) e.Kill();
            if (toKill.Count > 0) Puts($"BotController: Janitor killed {toKill.Count} legacy RL_Agent entities.");

            Puts("BotController: Server Initialized. Spawning 2 BasePlayer RL_Agents...");
            for (int i = 0; i < 2; i++)
            {
                SpawnBot(i);
            }

            // The "God Tree" Override: Dynamic Height Snap
            Vector3 treePos = new Vector3(102, 0, 100);
            treePos.y = TerrainMeta.HeightMap.GetHeight(treePos);
            var tree = GameManager.server.CreateEntity("assets/bundled/prefabs/autospawn/resource/v2_temp_forest/pine_a.prefab", treePos, Quaternion.identity);
            if (tree != null)
            {
                tree.Spawn();
                Puts($"BotController: 'God Tree' spawned at {treePos}");
            }

            timer.Every(0.1f, ProcessActions);
            timer.Every(3f, WakeAllBots);
        }

        private void SpawnBot(int idx)
        {
            float spawnX = 195.0f + (idx * 1.5f);
            float spawnZ = 145.0f;
            float spawnY = TerrainMeta.HeightMap.GetHeight(new Vector3(spawnX, 0, spawnZ));
            Vector3 spawnPos = new Vector3(spawnX, spawnY, spawnZ);

            // The Prefab Revert: Standard player for proper animation rig
            var bot = GameManager.server.CreateEntity(
                "assets/prefabs/player/player.prefab", 
                spawnPos, Quaternion.identity) as BasePlayer;

            if (bot == null)
            {
                Puts($"BotController: ERROR - Failed to create RL_Agent_{idx}!");
                return;
            }

            bot.Spawn();

            // Destroy the Parking Brake: Remove client-authoritative movement
            var moveComp = bot.GetComponent<PlayerWalkMovement>();
            if (moveComp != null) UnityEngine.Object.Destroy(moveComp);

            // Wake Up Immediately: Force out of sleeping state
            bot.EndSleeping();

            // Anti-Hack Bypass: Mark as Admin to prevent rubberband snapping
            bot.SetPlayerFlag(BasePlayer.PlayerFlags.IsAdmin, true);

            bot.InitializeHealth(99999f, 99999f);
            timer.Once(1f, () => { if (bot != null) bot.EndSleeping(); });
            bot.displayName = $"RL_Agent_{idx}";
            bot.health = 99999f;
            bot.metabolism.calories.value = 1000;
            bot.metabolism.hydration.value = 1000;

            // Equip the Rock: Give the bot a rock and force it into the active hand
            timer.Once(2f, () => {
                if (bot == null || bot.IsDestroyed) return;
                var rock = ItemManager.CreateByName("rock", 1);
                if (rock != null)
                {
                    rock.MoveToContainer(bot.inventory.containerBelt, 0);
                    bot.UpdateActiveItem(rock.uid);
                    bot.SendNetworkUpdateImmediate();
                    Puts($"BotController: RL_Agent_{idx} armed with Rock.");
                }
            });

            _bots.Add(bot);
            Puts($"BotController: RL_Agent_{idx} spawned at {spawnPos}");
        }

        private void ProcessActions()
        {
            // Sync with 10 FPS training loop
            for (int i = 0; i < _bots.Count; i++)
            {
                var bot = _bots[i];
                if (bot == null || bot.IsDead()) continue;

                // Persistence Locks
                bot.health = 99999f;
                bot.metabolism.calories.value = 1000;
                bot.metabolism.hydration.value = 1000;

                // Absolute path to shared-data action files (must match Python writer)
                string botActionsPath = $"C:/Projects/rust-rl-agent/shared-data/actions_{i}.json";
                
                if (!File.Exists(botActionsPath)) continue;

                try
                {
                    string json = File.ReadAllText(botActionsPath);
                    var actions = JsonConvert.DeserializeObject<Dictionary<string, object>>(json);

                    // 1. Physical Rotation (LookX/LookY)
                    float lookX = Convert.ToSingle(actions["LookX"]);
                    float lookY = Convert.ToSingle(actions["LookY"]);
                    bot.viewAngles = new Vector3(lookY, lookX, 0);
                    bot.transform.rotation = Quaternion.Euler(0, lookX, 0);

                    // 2. Puppeteer the Swing: Direct server-side melee attack + animation broadcast
                    bool wantsToAttack = Convert.ToBoolean(actions["Attack"]);
                    if (wantsToAttack)
                    {
                        BaseMelee melee = bot.GetActiveItem()?.GetHeldEntity() as BaseMelee;
                        if (melee != null) melee.ServerUse();
                        bot.SignalBroadcast(BaseEntity.Signal.Attack, string.Empty);
                    }

                    // 3. Puppeteer the Legs: Direct position translation + animation state
                    float moveX = Convert.ToSingle(actions["MoveX"]);
                    float moveZ = Convert.ToSingle(actions["MoveZ"]);
                    bool isSprinting = Convert.ToBoolean(actions["Sprint"]);
                    float moveSpeed = 5f * (isSprinting ? 1.4f : 1.0f);

                    // Set modelState for animation rendering (Bitwise flag manipulation)
                    bot.modelState.flags |= (int)ModelState.Flag.OnGround;
                    bot.modelState.flags &= ~(int)ModelState.Flag.Sprinting;
                    bot.modelState.flags &= ~(int)ModelState.Flag.Jumped;

                    if (Mathf.Abs(moveX) > 0.1f || Mathf.Abs(moveZ) > 0.1f)
                    {
                        Vector3 dir = (bot.transform.forward * moveZ + bot.transform.right * moveX).normalized;
                        Vector3 newPos = bot.transform.position + dir * moveSpeed * 0.1f; // Fixed 0.1s tick
                        newPos.y = TerrainMeta.HeightMap.GetHeight(newPos);
                        bot.MovePosition(newPos);
                        bot.TransformChanged();
                        
                        if (isSprinting) bot.modelState.flags |= (int)ModelState.Flag.Sprinting;
                    }

                    // 4. Jump override
                    if (Convert.ToBoolean(actions["Jump"]))
                    {
                        bot.MovePosition(bot.transform.position + Vector3.up * 1.5f);
                        bot.modelState.flags |= (int)ModelState.Flag.Jumped;
                    }

                    bot.SendNetworkUpdateImmediate();
                }
                catch (Exception)
                {
                    // Silent fail for lock-contention, retry next tick
                }
            }
        }

        private void WakeAllBots()
        {
            int count = 0;
            foreach (var bot in _bots)
            {
                if (bot == null) continue;
                if (bot.IsSleeping())
                {
                    bot.EndSleeping();
                    bot.SendNetworkUpdateImmediate();
                    count++;
                }
            }
            if (count > 0) Puts($"WakeAllBots: Woke up {count} sleeping agents.");
        }

        private void Unload() 
        { 
            foreach (var bot in _bots) 
            {
                if (bot != null && !bot.IsDestroyed) bot.Kill();
            }
        }
    }
}
