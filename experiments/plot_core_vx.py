import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INPUT = 'experiments/contact_trace.json'
OUT = 'experiments/plots/contact_core_vx.png'
OUT_Z = 'experiments/plots/contact_core_vx_zoom_250_320.png'
os.makedirs(os.path.dirname(OUT), exist_ok=True)

with open(INPUT,'r') as f:
    data = json.load(f)
recs = data.get('records', [])
steps = [r.get('step', i) for i, r in enumerate(recs)]
vx = [r.get('vx', 0.0) for r in recs]

plt.figure(figsize=(10,4))
plt.plot(steps, vx, linewidth=2, color='C0')
plt.xlabel('step')
plt.ylabel('vx (signed forward)')
plt.title('Core run vx')
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT)
print('Saved', OUT)

# zoom
lo, hi = 250, 320
idx = [i for i, s in enumerate(steps) if lo <= s <= hi]
if idx:
    s_steps = [steps[i] for i in idx]
    s_vx = [vx[i] for i in idx]
    plt.figure(figsize=(10,3))
    plt.plot(s_steps, s_vx, linewidth=2, color='C0')
    plt.xlabel('step')
    plt.ylabel('vx (signed forward)')
    plt.title(f'Core run vx zoom {lo}-{hi}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_Z)
    print('Saved', OUT_Z)
else:
    print('No data in range')
