#!/usr/bin/env python3
"""Compare stats when enemy_visible is True vs False from debug_step_logs_ep0.jsonl
Writes a summary to analysis_enemy_visibility.txt
"""
import json
from pathlib import Path
from collections import defaultdict
import math

IN = Path("debug_step_logs_ep0.jsonl")
OUT = Path("analysis_enemy_visibility.txt")

def safe_mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs)/len(xs) if xs else float('nan')

def main():
    stats = {True: defaultdict(list), False: defaultdict(list)}
    total = {True:0, False:0}
    if not IN.exists():
        print("Missing", IN)
        return
    with IN.open() as f:
        for line in f:
            d = json.loads(line)
            vis = bool(d.get('enemy_visible', False))
            total[vis] += 1
            # det_action and sampled_action
            for k in ('det_action','sampled_action','action'):
                v = d.get(k)
                if v is None:
                    continue
                # store per-dim
                for i,comp in enumerate(v):
                    stats[vis][f"{k}[{i}]"] .append(float(comp))
            # wall_distance, speed, gaze, base_reward, reward
            for k in ('wall_distance','speed','gaze','base_reward','reward'):
                stats[vis][k].append(d.get(k))
            # nearest_enemy_dist may be null
            ned = d.get('nearest_enemy_dist')
            stats[vis]['nearest_enemy_dist'].append(ned if ned is not None else None)
            # obs lidar: assume LiDAR in obs[5:17], front bins obs[5:8], back bin obs[16]
            obs = d.get('obs') or []
            if len(obs) >= 17:
                lidar = obs[5:17]
                front_min = min(lidar[0:3])
                back = lidar[11]
                stats[vis]['front_min'].append(float(front_min))
                stats[vis]['back'].append(float(back))
            else:
                stats[vis]['front_min'].append(None)
                stats[vis]['back'].append(None)

    with OUT.open('w') as o:
        o.write('Enemy visible comparison\n')
        o.write('========================\n\n')
        for vis in (False, True):
            o.write(f"Visible={vis}:\n")
            o.write(f"  Count: {total[vis]}\n")
            keys = sorted(stats[vis].keys())
            for k in keys:
                vals = stats[vis][k]
                # filter None
                num = len([x for x in vals if x is not None])
                mean = safe_mean([x for x in vals if x is not None])
                # compute std
                if num>1:
                    var = sum((x-mean)**2 for x in vals if x is not None)/(num-1)
                    sd = math.sqrt(var)
                else:
                    sd = float('nan')
                o.write(f"  {k}: n={num}, mean={mean:.6g}, sd={sd:.6g}\n")
            o.write('\n')
    print(f"Wrote summary to {OUT}")

if __name__ == '__main__':
    main()
