import os
import glob
import json
from tensorboard.backend.event_processing import event_accumulator

def analyze_ppo_morning(log_dir):
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    
    tags = ea.Tags()['scalars']
    target_tags = [
        'rollout/ep_rew_mean',
        'time/fps',
        'rollout/ep_len_mean',
        'custom/wood_count_avg'
    ]
    
    results = {}
    for tag in target_tags:
        if tag in tags:
            events = ea.Scalars(tag)
            last_event = events[-1]
            first_event = events[0]
            
            # Trend calculation
            recent_events = events[-50:]
            history = [e.value for e in recent_events]
            avg_recent = sum(history) / len(history) if history else 0
            
            results[tag] = {
                'value': last_event.value,
                'step': last_event.step,
                'first_value': first_event.value,
                'recent_avg': avg_recent,
                'history_min': min(history) if history else 0,
                'history_max': max(history) if history else 0
            }
        else:
            results[tag] = 'Not Found'
            
    # Max step
    max_step = 0
    for tag in tags:
        max_step = max(max_step, ea.Scalars(tag)[-1].step)
    results['total_timesteps'] = max_step
    
    return results

if __name__ == "__main__":
    log_path = r"C:\Projects\ml_logs\tensorboard_logs_v2\PPO_1"
    res = analyze_ppo_morning(log_path)
    print(json.dumps(res, indent=2))
