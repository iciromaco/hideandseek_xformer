#!/usr/bin/env python3
import os
import sys

import numpy as np

F = "debug_model_env_dump.npz" if len(sys.argv) < 2 else sys.argv[1]
if not os.path.exists(F):
    print("dump not found:", F)
    sys.exit(1)

data = np.load(F, allow_pickle=True)
model_out = data["model_out"]
obs_hist = data["obs_hist"]
ctrl = data["ctrl"]
vel = data["vel"]
pos = data["pos"]
reward = data["reward"]

print("file:", F)
print(
    "shapes: model_out",
    model_out.shape,
    "obs_hist",
    obs_hist.shape,
    "ctrl",
    ctrl.shape,
    "vel",
    vel.shape,
    "pos",
    pos.shape,
    "reward",
    reward.shape,
)


def show_vec_stats(name, arr):
    arr = np.asarray(arr, dtype=np.float32)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    print(f"-- {name} mean (first 8):", np.round(mean[:8], 4))
    print(f"-- {name} std  (first 8):", np.round(std[:8], 4))
    return mean, std


print("\nModel output stats:")
mo_mean, mo_std = show_vec_stats("model_out", model_out)
neg_frac = np.mean(model_out[:, 0] < 0.0)
print("fraction negative for model_out[:,0]:", float(neg_frac))

print("\nCtrl stats:")
show_vec_stats("ctrl", ctrl)

print("\nVel stats: mean/std", float(np.mean(vel)), float(np.std(vel)))
print("Reward stats: mean/std", float(np.mean(reward)), float(np.std(reward)))

print("\nObservation history stats (flattened across seq):")
ns, sl, od = obs_hist.shape
flat = obs_hist.reshape(ns * sl, od)
obs_mean = flat.mean(axis=0)
obs_std = flat.std(axis=0)
print("obs_dim:", od, "flattened samples:", flat.shape[0])
print("obs mean (first 16):", np.round(obs_mean[:16], 4))
print("obs std  (first 16):", np.round(obs_std[:16], 4))

# show top-k most varying obs dims
idx = np.argsort(-obs_std)[:12]
print("\nTop 12 obs dims by std:")
for i in idx:
    print(f"  dim {i}: mean={obs_mean[i]:.4f} std={obs_std[i]:.4f}")

# correlation between model_out[:,0] and reward
try:
    corr = np.corrcoef(model_out[:, 0], reward)[0, 1]
    print("\nCorr(model_out[:,0], reward)=", float(corr))
except Exception:
    print("\nCould not compute correlation")

print("\nSample frames:")
for i in range(min(8, model_out.shape[0])):
    print(f"FRAME {i}: model_out[:6]={np.round(model_out[i,:6],4)} ctrl[:6]={np.round(ctrl[i,:6],4)} vel={vel[i]:.3f} reward={reward[i]:.3f}")

print("\nWrote analysis summary to stdout")
