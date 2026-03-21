import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

hold_vals = ['-1.0','-0.5','-0.25','0.25','0.5','1.0']
jsons = [f'experiments/step_response_nowalls_h{v.replace("-","m").replace(".","p")}.json' for v in hold_vals]
labels = [f'h={v}' for v in hold_vals]

out_png = 'experiments/plots/compare_contact_batch_vx.png'
out_png_zoom = 'experiments/plots/compare_contact_batch_vx_zoom_250_320.png'
os.makedirs(os.path.dirname(out_png), exist_ok=True)

trim = 80
plt.figure(figsize=(10,6))
for j, lab in zip(jsons, labels):
    if not os.path.exists(j):
        continue
    with open(j,'r') as f:
        data = json.load(f)
    recs = data.get('records', [])
    vx = [r.get('vx', 0.0) for r in recs]
    if len(vx) <= trim:
        continue
    vx_trim = vx[trim:]
    t = np.arange(len(vx_trim)) + trim
    plt.plot(t, vx_trim, label=lab, linewidth=1)

# overlay contact_trace single-run vx if available
ct_path = 'experiments/contact_trace.json'
if os.path.exists(ct_path):
    with open(ct_path,'r') as f:
        cdata = json.load(f)
    crecs = cdata.get('records', [])
    c_vx = [r.get('vx', 0.0) for r in crecs]
    t_c = np.arange(len(c_vx))
    plt.plot(t_c, c_vx, label='contact_trace vx', color='k', linewidth=2.0, linestyle='--')

plt.xlabel('step')
plt.ylabel('vx')
plt.legend(ncol=2)
plt.grid(True)
plt.tight_layout()
plt.savefig(out_png)
print('Saved', out_png)

# zoom
lo, hi = 250, 320
plt.figure(figsize=(10,4))
for j, lab in zip(jsons, labels):
    if not os.path.exists(j):
        continue
    with open(j,'r') as f:
        data = json.load(f)
    recs = data.get('records', [])
    vx = [r.get('vx', 0.0) for r in recs]
    idx = [i for i in range(len(vx)) if lo <= i <= hi]
    if not idx: continue
    s_steps = [i for i in idx]
    s_vx = [vx[i] for i in idx]
    plt.plot(s_steps, s_vx, label=lab, linewidth=1)

if os.path.exists(ct_path):
    with open(ct_path,'r') as f:
        cdata = json.load(f)
    crecs = cdata.get('records', [])
    c_vx = [r.get('vx', 0.0) for r in crecs]
    idx = [i for i in range(len(c_vx)) if lo <= i <= hi]
    if idx:
        s_steps = [i for i in idx]
        s_vx = [c_vx[i] for i in idx]
        plt.plot(s_steps, s_vx, label='contact_trace vx', color='k', linewidth=2.0, linestyle='--')

plt.xlabel('step')
plt.ylabel('vx')
plt.title(f'Zoom steps {lo}-{hi}')
plt.legend(ncol=2)
plt.grid(True)
plt.tight_layout()
plt.savefig(out_png_zoom)
print('Saved', out_png_zoom)
