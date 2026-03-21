import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

cases = ['m1p0','m0p5','m0p25','0p25','0p5','0p1p0']
# but filenames created by batch use -1.0 -> m1p0 etc; ensure matching
hold_vals = ['-1.0','-0.5','-0.25','0.25','0.5','1.0']
jsons = [f'experiments/step_response_nowalls_h{v.replace("-","m").replace(".","p")}.json' for v in hold_vals]
labels = [f'h={v}' for v in hold_vals]

series = []
stats_list = []
for j in jsons:
    if not os.path.exists(j):
        series.append(None)
        stats_list.append(None)
        continue
    with open(j,'r') as f:
        data = json.load(f)
    recs = data.get('records', [])
    # use vx (body forward velocity) to be consistent across plots
    vals = [r.get('vx', 0.0) for r in recs]
    series.append(vals)
    stats_list.append(data.get('stats', {}))

# make plot
out_png = 'experiments/plots/step_response_nowalls_batch.png'
os.makedirs(os.path.dirname(out_png), exist_ok=True)
plt.figure(figsize=(10,5))
trim = 80
for vals, lab, st in zip(series, labels, stats_list):
    if vals is None: continue
    vals_trim = vals[trim:]
    t = np.arange(len(vals_trim)) + trim
    plt.plot(t, vals_trim, label=lab, linewidth=1)
plt.xlabel('step')
plt.ylabel('vx')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(out_png)
print('Saved', out_png)
