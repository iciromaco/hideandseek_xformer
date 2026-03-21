import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

hold_vals = ['-1.0','-0.5','-0.25','0.25','0.5','1.0']
jsons = [f'experiments/step_response_nowalls_h{v.replace("-","m").replace(".","p")}.json' for v in hold_vals]
labels = [f'h={v}' for v in hold_vals]

out_png = 'experiments/plots/local_vx_overlay_signed_all_cases.png'
out_png_zoom = 'experiments/plots/local_vx_overlay_signed_zoom_250_320.png'
trim = 80
os.makedirs(os.path.dirname(out_png), exist_ok=True)

plt.figure(figsize=(10,6))
for j, lab in zip(jsons, labels):
    if not os.path.exists(j):
        continue
    with open(j,'r') as f:
        data = json.load(f)
    recs = data.get('records', [])
    signed = [r.get('signed_forward', 0.0) for r in recs]
    if len(signed) <= trim:
        continue
    s = signed[trim:]
    t = np.arange(len(s)) + trim
    plt.plot(t, s, label=lab, linewidth=1)

# overlay single-run contact_trace signed_forward if present
ct = 'experiments/contact_trace.json'
if os.path.exists(ct):
    with open(ct,'r') as f:
        c = json.load(f)
    crecs = c.get('records', [])
    csig = [r.get('signed_forward', None) for r in crecs]
    if any(v is not None for v in csig):
        csig2 = [v if v is not None else 0.0 for v in csig]
        t_c = np.arange(len(csig2))
        plt.plot(t_c, csig2, label='contact_trace (single run)', color='k', linewidth=2.0, linestyle='--')

plt.xlabel('step')
plt.ylabel('local vx (signed_forward)')
plt.title('Local vx overlay (signed_forward) — trimmed warmup')
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
    signed = [r.get('signed_forward', 0.0) for r in recs]
    idx = [i for i in range(len(signed)) if lo <= i <= hi]
    if not idx: continue
    s_steps = [i for i in idx]
    s_v = [signed[i] for i in idx]
    plt.plot(s_steps, s_v, label=lab, linewidth=1)

if os.path.exists(ct):
    with open(ct,'r') as f:
        c = json.load(f)
    crecs = c.get('records', [])
    csig = [r.get('signed_forward', None) for r in crecs]
    idx = [i for i in range(len(csig)) if lo <= i <= hi]
    if idx:
        s_steps = [i for i in idx]
        s_v = [csig[i] if csig[i] is not None else 0.0 for i in idx]
        plt.plot(s_steps, s_v, label='contact_trace (single run)', color='k', linewidth=2.0, linestyle='--')

plt.xlabel('step')
plt.ylabel('local vx (signed_forward)')
plt.title(f'Local vx overlay zoom {lo}-{hi}')
plt.legend(ncol=2)
plt.grid(True)
plt.tight_layout()
plt.savefig(out_png_zoom)
print('Saved', out_png_zoom)
