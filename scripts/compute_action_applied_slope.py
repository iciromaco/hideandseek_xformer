#!/usr/bin/env python3
"""
Compute linear fit (slope/intercept) and Pearson r between policy action[:,0] and env applied_forward.
Saves per-step slope CSV for requested step (default 74) to diagnostics/slope_step{step}.csv
Usage: python scripts/compute_action_applied_slope.py diagnostics/bias_dump_update50.pt [step]
"""
import sys, os
from pathlib import Path
import torch
import numpy as np
from scipy import stats

if len(sys.argv) < 2:
    print('Usage: python scripts/compute_action_applied_slope.py <bias_dump.pt> [step]')
    sys.exit(2)

p = Path(sys.argv[1])
step_to_save = int(sys.argv[2]) if len(sys.argv) > 2 else 74
if not p.exists():
    print('File not found:', p)
    sys.exit(2)

data = torch.load(str(p), map_location='cpu', weights_only=False)
ra = data.get('rollout_actions')
info = data.get('info_buffer', [])
if ra is None:
    print('rollout_actions missing')
    sys.exit(2)

ra = ra.detach().cpu().numpy()
T, N, AD = ra.shape
policy0 = ra[:, :, 0]

applied = np.full((T, N), np.nan, dtype=np.float32)
for t in range(min(T, len(info))):
    it = info[t]
    if isinstance(it, dict):
        v = it.get('applied_forward', None)
    else:
        v = it
    try:
        arr = np.asarray(v, dtype=np.float32)
        if arr.ndim == 1:
            applied[t, :min(N, arr.shape[0])] = arr[:min(N, arr.shape[0])]
        elif arr.ndim == 2 and arr.shape[0] == N:
            applied[t, :] = arr[:, 0]
    except Exception:
        pass

mask = np.isfinite(applied)
if mask.sum() == 0:
    print('No applied_forward samples found')
    sys.exit(2)

pol_flat = policy0.ravel()[mask.ravel()]
app_flat = applied.ravel()[mask.ravel()]

# global linear fit using least squares
slope, intercept = np.polyfit(pol_flat, app_flat, 1)
pearson_r, pval = stats.pearsonr(pol_flat, app_flat)

print(f'Global samples: {mask.sum()}')
print(f'Global slope: {slope:.6f}, intercept: {intercept:.6f}, pearson r: {pearson_r:.6f} (p={pval:.2e})')

# per-step slopes where possible
slopes = []
counts = []
for t in range(T):
    m = mask[t]
    if m.sum() < 2:
        slopes.append(np.nan)
        counts.append(int(m.sum()))
        continue
    x = policy0[t][m]
    y = applied[t][m]
    a, b = np.polyfit(x, y, 1)
    slopes.append(float(a))
    counts.append(int(m.sum()))

slopes = np.array(slopes)
valid = np.isfinite(slopes)
print(f'Per-step slope stats: mean={np.nanmean(slopes):.6f}, median={np.nanmedian(slopes):.6f}, min={np.nanmin(slopes):.6f}, max={np.nanmax(slopes):.6f}')

# save CSV for requested step
outdir = Path('diagnostics')
outdir.mkdir(exist_ok=True)
step = step_to_save
csv_path = outdir / f'slope_step{step}.csv'
rows = []
if step < T:
    x = policy0[step]
    y = applied[step]
    for j in range(len(x)):
        rows.append((j, float(x[j]), float(y[j])))
    import csv
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['env','policy0','applied_forward'])
        w.writerows(rows)
    print(f'Saved step {step} scatter CSV to {csv_path}')
else:
    print(f'step {step} out of range (T={T})')

print('Done')
