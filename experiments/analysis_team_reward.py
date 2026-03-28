#!/usr/bin/env python3
import json
import math
from collections import Counter
from statistics import mean, pstdev

p = "experiments/team_reward_log.json"
with open(p, "r") as f:
    data = json.load(f)

N = len(data)

step_rewards = [rec.get("step_reward", 0.0) for rec in data]
team_from_compute = [rec.get("team_reward_from_compute", 0.0) for rec in data]

# team_reward_components may be null for many early steps
components = [rec.get("team_reward_components") for rec in data]
has_components = [c is not None for c in components]

# extract fields when present
dist_bonus = [c["dist_bonus"] for c in components if c]
dist_ratio = [c["dist_ratio"] for c in components if c]
min_seeker_dist = [c["min_seeker_dist"] for c in components if c]
seen_count = [c["seen_count"] for c in components if c]
learnable_seen = [c["learnable_hider_seen"] for c in components if c]

# info-level fields
applied_forward = [rec["info"].get("applied_forward", float("nan")) for rec in data]
agent_vx = [rec["info"].get("agent_vx", float("nan")) for rec in data]


def stats(arr):
    if not arr:
        return (0, 0, 0, 0)
    return (len(arr), mean(arr), min(arr), max(arr))


print(f"Total records: {N}")
print(f"Records with team_reward_components: {sum(has_components)} ({sum(has_components)/N:.2%})")

print("\n-- step_reward --")
cnt, m, mn, mx = stats(step_rewards)
sd = pstdev(step_rewards) if cnt > 1 else 0.0
print(f"count={cnt} mean={m:.6f} std={sd:.6f} min={mn:.6f} max={mx:.6f}")

print("\n-- team_reward_from_compute --")
cnt, m, mn, mx = stats(team_from_compute)
sd = pstdev(team_from_compute) if cnt > 1 else 0.0
print(f"count={cnt} mean={m:.6f} std={sd:.6f} min={mn:.6f} max={mx:.6f}")

print("\n-- dist_bonus (when present) --")
if dist_bonus:
    cnt, m, mn, mx = stats(dist_bonus)
    sd = pstdev(dist_bonus) if cnt > 1 else 0.0
    print(f"count={cnt} mean={m:.6f} std={sd:.6f} min={mn:.6f} max={mx:.6f}")
    # fraction > 0.01
    frac_pos = sum(1 for v in dist_bonus if v > 0.01) / cnt
    print(f"fraction >0.01 = {frac_pos:.2%}")
    # simple histogram
    buckets = [0] * 10
    for v in dist_bonus:
        bi = min(int(v * 10), 9)
        buckets[bi] += 1
    print("hist buckets (0-0.1,...,0.9-1.0):", buckets)
else:
    print("no dist_bonus present")

print("\n-- min_seeker_dist --")
if min_seeker_dist:
    cnt, m, mn, mx = stats(min_seeker_dist)
    sd = pstdev(min_seeker_dist) if cnt > 1 else 0.0
    print(f"count={cnt} mean={m:.6f} std={sd:.6f} min={mn:.6f} max={mx:.6f}")

print("\n-- seen_count --")
if seen_count:
    cnt = len(seen_count)
    ctr = Counter(seen_count)
    print(f"count={cnt} distribution={dict(ctr)}")

print("\n-- learnable_hider_seen --")
if learnable_seen:
    cnt = len(learnable_seen)
    trues = sum(1 for v in learnable_seen if v)
    print(f"count={cnt} true={trues} ({trues/cnt:.2%})")

print("\n-- applied_forward vs agent_vx correlation (Pearson) --")
# compute pearson for pairs without nan
pairs = [(a, v) for a, v in zip(applied_forward, agent_vx) if (not math.isnan(a)) and (not math.isnan(v))]
if len(pairs) >= 2:
    A = [p[0] for p in pairs]
    V = [p[1] for p in pairs]
    meanA = mean(A)
    meanV = mean(V)
    cov = sum((x - meanA) * (y - meanV) for x, y in zip(A, V)) / len(A)
    stdA = math.sqrt(sum((x - meanA) ** 2 for x in A) / len(A))
    stdV = math.sqrt(sum((y - meanV) ** 2 for y in V) / len(V))
    corr = cov / (stdA * stdV) if stdA > 0 and stdV > 0 else float("nan")
    print(f"pairs={len(A)} pearson={corr:.6f}")
else:
    print("not enough pairs to compute correlation")

# top 5 largest dist_bonus with time
if dist_bonus:
    indexed = [(i, c["dist_bonus"]) for i, c in enumerate(components) if c]
    top5 = sorted(indexed, key=lambda x: x[1], reverse=True)[:5]
    print("\nTop 5 dist_bonus (index, value):")
    for idx, val in top5:
        print(idx, val)

print("\nDone.")
