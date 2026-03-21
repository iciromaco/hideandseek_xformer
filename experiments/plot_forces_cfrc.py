import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INPUT = 'experiments/contact_trace.json'
OUT_PNG = 'experiments/plots/contact_forces_cfrc.png'
OUT_PNG_ZOOM = 'experiments/plots/contact_forces_cfrc_zoom_250_320.png'
os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)

with open(INPUT, 'r') as f:
    data = json.load(f)
recs = data.get('records', [])
steps = [r['step'] for r in recs]
# actuator forces: may be None or list
act = [r.get('actuator_forces') for r in recs]
cfrc = [r.get('cfrc_body') for r in recs]

# prepare actuator series for first 4 actuators (if present)
max_act = 0
for a in act:
    if a is not None:
        max_act = max(max_act, len(a))
act_series = []
for i in range(max_act):
    series = [ (a[i] if (a is not None and i < len(a)) else np.nan) for a in act ]
    act_series.append(series)

# prepare cfrc components (6)
cfrc_series = []
for j in range(6):
    series = [ (c[j] if (c is not None and len(c) >= 6) else np.nan) for c in cfrc ]
    cfrc_series.append(series)

# overall plot
plt.figure(figsize=(12,8))
plt.subplot(2,1,1)
for i, s in enumerate(act_series):
    plt.plot(steps, s, label=f'act_{i}', linewidth=1)
plt.ylabel('actuator_force')
plt.legend(ncol=4)
plt.grid(True)

plt.subplot(2,1,2)
for j, s in enumerate(cfrc_series):
    plt.plot(steps, s, label=f'cfrc_{j}', linewidth=1)
plt.ylabel('cfrc_body')
plt.xlabel('step')
plt.legend(ncol=3)
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_PNG)
print('Saved', OUT_PNG)

# zoomed region
lo, hi = 250, 320
idx = [i for i, s in enumerate(steps) if lo <= s <= hi]
if idx:
    s_steps = [steps[i] for i in idx]
    plt.figure(figsize=(12,6))
    plt.subplot(2,1,1)
    for i, series in enumerate(act_series):
        s = [series[k] for k in idx]
        plt.plot(s_steps, s, label=f'act_{i}', linewidth=2 if i==0 else 1)
    plt.ylabel('actuator_force')
    plt.legend(ncol=4)
    plt.grid(True)

    plt.subplot(2,1,2)
    for j, series in enumerate(cfrc_series):
        s = [series[k] for k in idx]
        plt.plot(s_steps, s, label=f'cfrc_{j}', linewidth=2 if j==0 else 1)
    plt.ylabel('cfrc_body')
    plt.xlabel('step')
    plt.legend(ncol=3)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_PNG_ZOOM)
    print('Saved', OUT_PNG_ZOOM)
else:
    print('No points in zoom range')
