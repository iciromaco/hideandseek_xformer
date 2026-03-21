import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INPUT = 'experiments/contact_trace.json'
OUT_PNG = 'experiments/plots/contact_trace.png'
OUT_PNG_ZOOM = 'experiments/plots/contact_trace_zoom_250_320.png'

os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
with open(INPUT, 'r') as f:
    data = json.load(f)
records = data.get('records', [])
steps = [r['step'] for r in records]
vx = [r.get('vx', 0.0) for r in records]
applied = [r.get('applied', 0.0) for r in records]
qv0 = [(r.get('qvel_slice') or [None])[0] for r in records]
act0 = [(r.get('actuator_forces') or [None])[0] for r in records]

plt.figure(figsize=(10,6))
plt.subplot(3,1,1)
plt.plot(steps, vx, label='vx (joint)', color='C0', linewidth=2)
plt.plot(steps, qv0, label='qvel_slice[0]', color='k', linestyle='--', alpha=0.8)
plt.ylabel('vx')
plt.legend()
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(steps, applied, label='applied_forward', color='C3')
plt.ylabel('applied')
plt.legend()
plt.grid(True)

plt.subplot(3,1,3)
plt.plot(steps, act0, label='actuator_forces[0]')
plt.ylabel('act0')
plt.xlabel('step')
plt.legend()
plt.grid(True)

# mark interesting region
plt.tight_layout()
plt.savefig(OUT_PNG)
print('Saved', OUT_PNG)

# --- zoomed view around suspicious region
try:
    lo, hi = 250, 320
    idx = [i for i, s in enumerate(steps) if lo <= s <= hi]
    if idx:
        s_steps = [steps[i] for i in idx]
        s_vx = [vx[i] for i in idx]
        s_qv0 = [qv0[i] for i in idx]
        s_ap = [applied[i] for i in idx]
        plt.figure(figsize=(10,4))
        plt.plot(s_steps, s_vx, label='vx (joint)', color='C0', linewidth=2)
        plt.plot(s_steps, s_qv0, label='qvel_slice[0]', color='k', linestyle='--')
        # autoscale with margin
        arr = np.array(s_vx)
        if arr.size:
            mn, mx = np.nanmin(arr), np.nanmax(arr)
            rng = max(abs(mx - mn), 1e-6)
            plt.ylim(mn - 0.2 * rng, mx + 0.2 * rng)
        plt.xlabel('step')
        plt.ylabel('vx')
        plt.title(f'Zoom steps {lo}-{hi}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(OUT_PNG_ZOOM)
        print('Saved', OUT_PNG_ZOOM)
except Exception:
    pass
