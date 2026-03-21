#!/usr/bin/env python3
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

F = "debug_model_env_dump_ext.npz" if len(sys.argv) < 2 else sys.argv[1]
out_dir = "plots"
os.makedirs(out_dir, exist_ok=True)

data = np.load(F, allow_pickle=True)
model_out = data["model_out"]
ctrl = data["ctrl"]
reward = data["reward"]
team_rb = data["team_rb"] if "team_rb" in data else None
team_min_seeker_dist = data["team_min_seeker_dist"] if "team_min_seeker_dist" in data else None
team_gaze_max = data["team_gaze_max"] if "team_gaze_max" in data else None

T = model_out.shape[0]
ts = np.arange(T)

plt.figure(figsize=(10, 6))
plt.plot(ts, model_out[:, 0], label="model_out[0]")
plt.plot(ts, ctrl[:, 0], label="env.ctrl[0]")
plt.plot(ts, reward, label="reward")
if team_rb is not None:
    plt.plot(ts, team_rb, label="team_rb", linestyle="--")
plt.legend()
plt.title("Model forward output, ctrl[0], reward, team_rb")
plt.xlabel("frame")
plt.grid(True)
path1 = os.path.join(out_dir, "timeseries_model_ctrl_reward.png")
plt.savefig(path1, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 6))
if team_min_seeker_dist is not None:
    plt.plot(ts, team_min_seeker_dist, label="team_min_seeker_dist")
if team_gaze_max is not None:
    plt.plot(ts, team_gaze_max, label="team_gaze_max")
plt.legend()
plt.title("Team metrics")
plt.xlabel("frame")
plt.grid(True)
path2 = os.path.join(out_dir, "team_metrics.png")
plt.savefig(path2, bbox_inches="tight")
plt.close()

# correlation between model_out[:,0] and team_rb / reward
out_lines = []
if team_rb is not None:
    corr_rb = np.corrcoef(model_out[:, 0], team_rb)[0, 1]
    out_lines.append(f"corr(model_out0, team_rb) = {corr_rb:.4f}")
else:
    out_lines.append("team_rb not in dump")
corr_reward = np.corrcoef(model_out[:, 0], reward)[0, 1]
out_lines.append(f"corr(model_out0, reward) = {corr_reward:.4f}")

with open(os.path.join(out_dir, "correlations.txt"), "w") as f:
    f.write("\n".join(out_lines) + "\n")

print("wrote", path1, path2, "and correlations.txt")
