import sys
import torch
import numpy as np
from pathlib import Path

def summarize_tensor(t):
    try:
        a = t.detach().cpu().numpy()
        return dict(shape=a.shape, mean=float(np.mean(a)), min=float(np.min(a)), max=float(np.max(a)))
    except Exception:
        return str(type(t))


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/inspect_bias_dump.py diagnostics/bias_dump_updateX.pt")
        return 2
    p = Path(sys.argv[1])
    if not p.exists():
        print(f"File not found: {p}")
        return 2
    data = torch.load(str(p), map_location='cpu')
    if not isinstance(data, dict):
        print(f"Loaded object type: {type(data)}")
        return 0
    print(f"Loaded dump: {p.name}\nKeys:")
    for k in sorted(list(data.keys())):
        v = data[k]
        if torch.is_tensor(v):
            s = summarize_tensor(v)
            print(f" - {k}: TENSOR {s}")
        elif isinstance(v, (list, tuple)):
            print(f" - {k}: {type(v).__name__} len={len(v)}")
        elif isinstance(v, dict):
            print(f" - {k}: dict keys={list(v.keys())[:10]}")
        else:
            print(f" - {k}: {type(v).__name__} -> {str(v)[:200]}")

    # If rollout_actions present, print small sample
    if 'rollout_actions' in data:
        try:
            ra = data['rollout_actions']
            if torch.is_tensor(ra):
                a = ra.detach().cpu().numpy()
                print('\nrollout_actions sample stats:')
                print(' shape:', a.shape)
                print(' mean:', float(np.mean(a)))
                print(' mean axis (0):', np.mean(a, axis=(0,1)).tolist() if a.ndim>=3 else np.mean(a, axis=0).tolist())
                print(' min:', float(np.min(a)), 'max:', float(np.max(a)))
        except Exception as e:
            print('rollout_actions inspect failed:', e)

    # If state_dict present, list top-level parameter names and norms
    if 'state_dict' in data and isinstance(data['state_dict'], dict):
        print('\nstate_dict param summary:')
        sd = data['state_dict']
        count = 0
        for k, v in sd.items():
            if torch.is_tensor(v):
                try:
                    arr = v.detach().cpu().numpy()
                    norm = float(np.linalg.norm(arr.ravel()))
                    print(f"  {k}: shape={arr.shape} norm={norm:.4e}")
                except Exception:
                    print(f"  {k}: tensor (cannot summarize)")
            else:
                print(f"  {k}: {type(v).__name__}")
            count += 1
            if count >= 20:
                print('  ...')
                break

if __name__ == '__main__':
    raise SystemExit(main())
