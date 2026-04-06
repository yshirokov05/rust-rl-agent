import numpy as np
import os
import glob
import time

LOG_BASE = r"C:\Projects\rust-rl-agent\shared-data\depth_logs"

def audit_swarm():
    print(f"{'Bot_ID':<8} | {'Delta':<15} | {'Action':<20} | {'Status':<10}")
    print("-" * 60)
    
    # 4 procs, each has 4 bots. 
    # proc_0: 0,1,2,3
    # proc_1: 4,5,6,7
    # proc_2: 8,9,10,11
    # proc_3: 12,13,14,15
    
    proc_map = {
        0: [0,1,2,3],
        1: [4,5,6,7],
        2: [8,9,10,11],
        3: [12,13,14,15]
    }
    
    for p_id, bots in proc_map.items():
        files = glob.glob(os.path.join(LOG_BASE, f"proc_{p_id}", "batch_*.npz"))
        if not files:
            for b in bots:
                print(f"Bot_{b:<4} | No Data         | N/A                  | INACTIVE")
            continue
            
        latest = max(files, key=os.path.getctime)
        data = np.load(latest)
        
        # In a batch of 200, we have data for all 4 bots? 
        # No, inference_v3.py: 
        # for bot_id in bots: 
        #   batch_buffer.append(...)
        # So a batch of 200 contains 200 interleaved steps (50 steps per bot).
        
        obs = data["obs"] # Shape (200, n_obs)
        actions = data["action"]
        
        for b_idx, b_id in enumerate(bots):
            # Extract steps for this specific bot
            # b_idx 0 is steps 0, 4, 8...
            # b_idx 1 is steps 1, 5, 9...
            bot_obs = obs[b_idx::4]
            bot_acts = actions[b_idx::4]
            
            if len(bot_obs) < 2:
                print(f"Bot_{b_id:<4} | Too few steps  | N/A                  | WAIT")
                continue
                
            p1 = bot_obs[-1][0:3]
            p2 = bot_obs[-2][0:3]
            delta = np.linalg.norm(p1 - p2)
            
            act = bot_acts[-1]
            act_str = f"F:{act[0]:.1f} S:{act[1]:.1f} A:{act[6]:.1f}"
            
            status = "ACTIVE"
            if delta < 0.01:
                # We need to know the TIME delta. 
                # 50 steps at 25Hz = 2 seconds. 
                # If delta is 0 for the whole batch, it's definitely stuck.
                status = "STUCK"
                
            print(f"Bot_{b_id:<4} | {delta:<15.4f} | {act_str:<20} | {status}")

if __name__ == "__main__":
    audit_swarm()
