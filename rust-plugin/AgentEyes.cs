using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json;
using UnityEngine;

namespace Carbon.Plugins
{
    [Info("AgentEyes", "Antigravity", "0.1.0")]
    [Description("Extracts tree and ore locations for an AI agent.")]
    public class AgentEyes : CarbonPlugin
    {
        private const string DataPath = "../../../../../../shared-data/vision.json";

        private void Init()
        {
            timer.Every(0.1f, WriteVisionData);
        }

        private void WriteVisionData()
        {
            var players = BasePlayer.activePlayerList;
            if (players.Count == 0) return;

            // For simplicity, we assume the first active player is our agent.
            var player = players[0];
            var playerPos = player.transform.position;

            var visionData = new VisionData
            {
                PlayerPosition = new Vector3Data(playerPos),
                NearestTree = GetNearestEntityVector(playerPos, "tree"),
                NearestOre = GetNearestEntityVector(playerPos, "ore")
            };

            var json = JsonConvert.SerializeObject(visionData, Formatting.Indented);
            File.WriteAllText(DataPath, json);
        }

        private Vector3Data GetNearestEntityVector(Vector3 origin, string type)
        {
            BaseEntity nearest = null;
            float minDistance = float.MaxValue;

            foreach (var entity in BaseEntity.serverEntities)
            {
                if (entity == null) continue;

                bool match = false;
                if (type == "tree" && (entity.ShortPrefabName.Contains("tree") || entity is TreeEntity)) match = true;
                if (type == "ore" && (entity.ShortPrefabName.Contains("ore") || entity.ShortPrefabName.Contains("resource"))) match = true;

                if (match)
                {
                    float dist = Vector3.Distance(origin, entity.transform.position);
                    if (dist < minDistance)
                    {
                        minDistance = dist;
                        nearest = entity;
                    }
                }
            }

            if (nearest == null) return null;

            // Calculate relative vector (x, y, z)
            Vector3 relative = nearest.transform.position - origin;
            return new Vector3Data(relative);
        }

        public class VisionData
        {
            public Vector3Data PlayerPosition { get; set; }
            public Vector3Data NearestTree { get; set; }
            public Vector3Data NearestOre { get; set; }
        }

        public class Vector3Data
        {
            public float X { get; set; }
            public float Y { get; set; }
            public float Z { get; set; }

            public Vector3Data(Vector3 v)
            {
                X = v.x;
                Y = v.y;
                Z = v.z;
            }
        }
    }
}
