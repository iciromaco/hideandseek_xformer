#!/usr/bin/env python3
"""
Compute correlation and sign-agreement between policy action[:,0] and env applied_forward
loads latest diagnostics/bias_dump_update*.pt
Usage: python scripts/calc_action_applied_corr.py
"""
import glob, re, torch, numpy as np, sys

path = None
if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    files = glob.glob('diagnostics/bias_dump_update*.pt')
    pairs = []
    for f in files:
        m = re.search(r'bias_dump_update(\d+)\\.pt$', f)
        if m:
            pairs.append((int(m.group(1)), f))
    if not pairs:
        print('No bias_dump files found in diagnostics (run from project root or pass path)', file=sys.stderr)
        sys.exit(2)
    pairs.sort()
    path = pairs[-1][1]
    print('Using', path)

# allow loading full dump (may require allowlist for numpy reconstruct)
try:
    import numpy as _np
    try:
        torch.serialization.add_safe_globals([_np._core.multiarray._reconstruct])
    except Exception:
        # fallback name
        try:
            torch.serialization.add_safe_globals([_np._core.multiarray._reconstruct])
        except Exception:
            pass
except Exception:
    pass

data = torch.load(path, map_location='cpu', weights_only=False)
if 'rollout_actions' not in data:
    print('rollout_actions not found in dump', file=sys.stderr)
    sys.exit(2)

ra = data['rollout_actions']
if not torch.is_tensor(ra):
    print('rollout_actions is not a tensor', file=sys.stderr)
    sys.exit(2)

ra = ra.detach().cpu().numpy()  # (T, N, act_dim)
T, N, AD = ra.shape
info = data.get('info_buffer', [])
# build applied_forward array
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
            # take first col if 2D
            applied[t, :] = arr[:, 0]
    except Exception:
        continue

mask = np.isfinite(applied)
if mask.sum() < 2:
    print('Not enough applied_forward samples', file=sys.stderr)
    sys.exit(2)

x = ra[:, :, 0].ravel()[mask.ravel()]
y = applied.ravel()[mask.ravel()]

mean_x, mean_y = float(x.mean()), float(y.mean())
num = float(((x - mean_x) * (y - mean_y)).sum())
den = float(np.sqrt(((x - mean_x) ** 2).sum() * ((y - mean_y) ** 2).sum()))
pearson_r = num / den if den != 0 else float('nan')
sign_agree = float(np.mean(np.sign(x) == np.sign(y)))

print(f'Overall samples: {mask.sum()} (T={T}, N={N})')
print(f'Overall Pearson r: {pearson_r:.6f}')
print(f'Sign agreement fraction: {sign_agree:.6f}')

# per-step stats for action0
mean_per_step = ra[:, :, 0].mean(axis=1)
min_idx = int(np.argmin(mean_per_step))
max_idx = int(np.argmax(mean_per_step))
print(f'min mean_action0 step: {min_idx} value={mean_per_step.min():.6f}')
print(f'max mean_action0 step: {max_idx} value={mean_per_step.max():.6f}')

# print top-10 most negative steps with sign-agree and per-step pearson if possible
from scipy.stats import pearsonr
per_step_r = []
per_step_sign = []
for t in range(T):
    x_t = ra[t, :, 0]
    y_t = applied[t, :]
    mask_t = np.isfinite(y_t)
    if mask_t.sum() < 2:
        per_step_r.append(float('nan'))
        per_step_sign.append(float('nan'))
        continue
    try:
        r_t, _ = pearsonr(x_t[mask_t], y_t[mask_t])
    except Exception:
        r_t = float('nan')
    per_step_r.append(float(r_t))
    per_step_sign.append(float(np.mean(np.sign(x_t[mask_t]) == np.sign(y_t[mask_t]))))

idxs = np.argsort(mean_per_step)
print('\nTop 10 most negative mean_action0 steps:')
for i in idxs[:10]:
    print(f' step {int(i)} mean={mean_per_step[i]:.6f} r={per_step_r[i]} sign_agree={per_step_sign[i]}')

print('\nDone')
