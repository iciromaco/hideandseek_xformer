import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

FILES = [
    ('step_response', 'experiments/step_response.json'),
    ('step_response_nowalls', 'experiments/step_response_nowalls.json'),
    ('step_response_sweep', 'experiments/step_response_sweep.json'),
]

# number of initial steps to trim (warmup / free-fall)
WARMUP_STEPS = 80

out_dir = os.path.join('experiments', 'plots')
os.makedirs(out_dir, exist_ok=True)

for name, path in FILES:
    if not os.path.exists(path):
        print('missing', path)
        continue
    with open(path, 'r') as f:
        data = json.load(f)
    records = data.get('records', [])
    if not records:
        print('no records for', name)
        continue
    steps = [r.get('step') for r in records]
    signed = [r.get('signed_forward', np.nan) for r in records]
    applied = [r.get('applied', np.nan) for r in records]

    # trim warmup steps
    if len(steps) > WARMUP_STEPS:
        steps = steps[WARMUP_STEPS:]
        signed = signed[WARMUP_STEPS:]
        applied = applied[WARMUP_STEPS:]

    plt.figure(figsize=(10,4))
    ax1 = plt.gca()
    ax1.plot(steps, signed, label='signed_forward', color='tab:blue')
    ax1.set_ylabel('signed_forward (m/s)')
    ax1.set_xlabel('step')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.step(steps, applied, where='post', label='applied', color='tab:orange', alpha=0.7)
    ax2.set_ylabel('applied action')
    ax2.set_ylim(min(applied) - 0.1, max(applied) + 0.1)

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title(name)
    outp = os.path.join(out_dir, f'{name}.png')
    plt.tight_layout()
    plt.savefig(outp)
    plt.close()
    # recompute steady stats after trimming and print for verification
    arr_signed = np.array(signed, dtype=np.float32)
    steady_last = min(100, len(arr_signed)//2) if len(arr_signed) > 0 else 0
    if steady_last > 0:
        steady_mean = float(arr_signed[-steady_last:].mean())
        steady_std = float(arr_signed[-steady_last:].std())
    else:
        steady_mean = 0.0; steady_std = 0.0
    print('Saved', outp, 'trimmed_warmup', WARMUP_STEPS, 'steady_mean', steady_mean, 'std', steady_std)

# For sweep, if blocks exist, make summary plot of block means if available
sweep_path = 'experiments/step_response_sweep.json'
if os.path.exists(sweep_path):
    with open(sweep_path,'r') as f:
        sdata = json.load(f)
    blocks = sdata.get('blocks') or sdata.get('records')
    # Try to plot each block's signed_forward if blocks is list of lists
    if isinstance(blocks, list) and blocks and isinstance(blocks[0], dict) and 'records' in blocks[0]:
        plt.figure(figsize=(10,4))
        for i, b in enumerate(blocks):
            recs = b.get('records', [])
            steps = [r.get('step') for r in recs]
            signed = [r.get('signed_forward', np.nan) for r in recs]
            if len(steps) > WARMUP_STEPS:
                steps = steps[WARMUP_STEPS:]
                signed = signed[WARMUP_STEPS:]
            plt.plot(steps, signed, label=f'block{i}')
        plt.legend(); plt.title('sweep blocks signed_forward')
        fn = os.path.join(out_dir, 'step_response_sweep_blocks.png')
        plt.savefig(fn); plt.close(); print('Saved', fn)
print('Done')
