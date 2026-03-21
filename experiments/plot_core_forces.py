import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INPUT = 'experiments/contact_trace.json'
OUT = 'experiments/plots/contact_core_xfrc.png'
OUT_Z = 'experiments/plots/contact_core_xfrc_zoom_250_320.png'
os.makedirs(os.path.dirname(OUT), exist_ok=True)

with open(INPUT,'r') as f:
    data = json.load(f)
recs = data.get('records', [])
steps = [r.get('step', i) for i, r in enumerate(recs)]
# collect xfrc_applied (6) per record
components = [[] for _ in range(6)]
for r in recs:
    x = r.get('xfrc_applied')
    if x and len(x) >= 6:
        for i in range(6):
            components[i].append(x[i])
    else:
        for i in range(6):
            components[i].append(np.nan)

labels = ['Fx','Fy','Fz','Mx','My','Mz']
plt.figure(figsize=(10,5))
for i in range(6):
    plt.plot(steps, components[i], label=labels[i])
plt.xlabel('step')
plt.ylabel('xfrc_applied (world)')
plt.legend(ncol=3)
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT)
print('Saved', OUT)

# zoom
lo, hi = 250, 320
idx = [i for i, s in enumerate(steps) if lo <= s <= hi]
if idx:
    s_steps = [steps[i] for i in idx]
    plt.figure(figsize=(10,4))
    for i in range(6):
        s_comp = [components[i][j] for j in idx]
        plt.plot(s_steps, s_comp, label=labels[i])
    plt.xlabel('step')
    plt.ylabel('xfrc_applied (world)')
    plt.title(f'Zoom steps {lo}-{hi}')
    plt.legend(ncol=3)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_Z)
    print('Saved', OUT_Z)
else:
    print('No data in range')
