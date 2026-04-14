import sys
import torch
import glob
import re
import numpy as np

if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    files = glob.glob('diagnostics/bias_dump_update*.pt')
    pairs = []
    for f in files:
        m = re.search(r'bias_dump_update(\d+)\\.pt$', f)
        if m:
            pairs.append((int(m.group(1)), f))
    if not pairs:
        print('No bias_dump files found')
        sys.exit(2)
    pairs.sort()
    path = pairs[-1][1]

print('Inspecting:', path)
data = torch.load(path, map_location='cpu', weights_only=False)
ra = data.get('rollout_actions')
if ra is None:
    print('rollout_actions not found in dump')
    sys.exit(2)

if not torch.is_tensor(ra):
    print('rollout_actions present but not tensor, type=', type(ra))
    sys.exit(2)

arr = ra.detach().cpu().numpy()
# arr expected shape (T, num_envs, act_dim)
T = arr.shape[0]
print('T, num_envs, act_dim =', arr.shape)
print('Per-step stats for action0 (first 40 steps):')
for t in range(min(40, T)):
    step = arr[t,:,0]
    print(f'step{t}: mean={float(np.mean(step)):.6f} min={float(np.min(step)):.6f} max={float(np.max(step)):.6f}')

# overall stats
print('\nOverall action means (axis TxN):', np.mean(arr, axis=(0,1)).tolist())

# if info_buffer contains applied_forward, print its mean per step if possible
info = data.get('info_buffer', None)
if info:
    # info is likely a list of length T
    print('\ninfo_buffer length:', len(info))
    af_list = []
    for i, item in enumerate(info):
        if isinstance(item, dict):
            v = item.get('applied_forward', None)
            af_list.append(v)
        else:
            af_list.append(None)
    print('sample applied_forward (first 20):', af_list[:20])

print('\nDone')
