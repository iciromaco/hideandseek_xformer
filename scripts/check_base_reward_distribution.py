#!/usr/bin/env python3
import json
from collections import Counter
from pathlib import Path

IN = Path("debug_step_logs_ep0.jsonl")


def count_visible_from_obs(obs):
    if obs is None:
        return 0
    L = len(obs)
    if L <= 17:
        return 0
    rem = L - 17
    n_others = rem // 8
    cnt = 0
    for i in range(n_others):
        vis_idx = 17 + i * 8 + 7
        try:
            if float(obs[vis_idx]) > 0.5:
                cnt += 1
        except Exception:
            continue
    return cnt


def main():
    if not IN.exists():
        print("Missing", IN)
        return
    total = 0
    vistrue = 0
    br_counter = Counter()
    viscount_counter = Counter()
    rows = []
    with IN.open() as f:
        for line in f:
            obj = json.loads(line)
            total += 1
            ev = bool(obj.get("enemy_visible", False))
            base = obj.get("base_reward")
            if ev:
                vistrue += 1
                br_counter[base] += 1
            # compute visible count from next_obs
            next_obs = obj.get("next_obs")
            vcnt = count_visible_from_obs(next_obs)
            viscount_counter[vcnt] += 1
            rows.append((ev, base, vcnt))

    print("Total frames:", total)
    print("enemy_visible True frames:", vistrue)
    print("\nBase reward counts for frames with enemy_visible True:")
    for k, v in br_counter.items():
        print(f"{k}: {v}")
    print("\nDistribution of visible-other counts (from next_obs):")
    for k in sorted(viscount_counter.keys()):
        print(f"{k}: {viscount_counter[k]}")

    # show mapping of base_reward to visible counts for enemy_visible True
    mapping = {}
    for ev, base, vcnt in rows:
        if ev:
            mapping.setdefault(base, []).append(vcnt)
    print("\nVisible counts per base_reward (sample):")
    for b, lst in mapping.items():
        from statistics import mean

        print(f"base={b}: n={len(lst)}, mean_visible={mean(lst):.3f}, unique_counts={sorted(set(lst))[:10]}")


if __name__ == "__main__":
    main()
