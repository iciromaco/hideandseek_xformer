import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

hold_vals = ['-1.0','-0.5','-0.25','0.25','0.5','1.0']
jsons = [f'experiments/step_response_nowalls_h{v.replace("-","m").replace(".","p")}.json' for v in hold_vals]
labels = [f'h={v}' for v in hold_vals]

out_png = 'experiments/plots/step_response_nowalls_vx_batch.png'
os.makedirs(os.path.dirname(out_png), exist_ok=True)

plt.figure(figsize=(10,5))
for j, lab in zip(jsons, labels):
    if not os.path.exists(j):
        print('missing', j)
        continue
    with open(j,'r') as f:
        data = json.load(f)
    recs = data.get('records', [])
    vx = [r.get('vx', 0.0) for r in recs]
    t = np.arange(len(vx))
    plt.plot(t, vx, label=lab, linewidth=1)

plt.xlabel('step')
plt.ylabel('vx')
plt.legend(ncol=2)
plt.grid(True)
plt.tight_layout()
plt.savefig(out_png)
print('Saved', out_png)
