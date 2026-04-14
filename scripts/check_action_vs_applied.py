#!/usr/bin/env python3
"""
Check elementwise difference between policy action[:,0] and env applied_forward
Usage: python scripts/check_action_vs_applied.py diagnostics/bias_dump_update50.pt
"""
import sys, torch, numpy as np
from pathlib import Path

if len(sys.argv) < 2:
    print('Usage: python scripts/check_action_vs_applied.py <bias_dump.pt>')
    sys.exit(2)

p = Path(sys.argv[1])
if not p.exists():
    print('File not found:', p)
    sys.exit(2)

# allow numpy reconstruct safe global if needed
try:
    import numpy as _np
    try:
        torch.serialization.add_safe_globals([_np._core.multiarray._reconstruct])
    except Exception:
        pass
except Exception:
    pass

data = torch.load(str(p), map_location='cpu', weights_only=False)
ra = data.get('rollout_actions')
if ra is None or not torch.is_tensor(ra):
    print('rollout_actions missing or not tensor')
    sys.exit(2)
ra = ra.detach().cpu().numpy()  # (T, N, act_dim)
T, N, AD = ra.shape
info = data.get('info_buffer', [])
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

policy0 = ra[:, :, 0]
# flatten masked
pol_flat = policy0.ravel()[mask.ravel()]
app_flat = applied.ravel()[mask.ravel()]
abs_diff = np.abs(pol_flat - app_flat)
max_abs = float(np.max(abs_diff))
mean_abs = float(np.mean(abs_diff))
median_abs = float(np.median(abs_diff))
exact_eq_frac = float(np.mean(abs_diff <= 1e-6))

print(f'Total samples compared: {mask.sum()}')
print(f'max abs diff: {max_abs:.6e}')
print(f'mean abs diff: {mean_abs:.6e}')
print(f'median abs diff: {median_abs:.6e}')
print(f'fraction exactly equal (<=1e-6): {exact_eq_frac:.6f}')

# show top mismatches
idx_sorted = np.argsort(-abs_diff)
print('\nTop 10 mismatches (abs diff, policy0, applied):')
for i in idx_sorted[:10]:
    print(i, abs_diff[i], pol_flat[i], app_flat[i])

# show step 74 samples if within range
step = 74
if step < T:
    pol_step = policy0[step]
    app_step = applied[step]
    print(f'\nStep {step} sample (first 10 envs):')
    for j in range(min(10, N)):
        vpol = pol_step[j]
        vapp = app_step[j]
        print(f'env{j}: policy0={vpol:.6f} applied={vapp}')

print('\nDone')
