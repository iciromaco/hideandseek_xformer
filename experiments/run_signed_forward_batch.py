import os
import sys
import json
import numpy as np

# ensure project root on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments.run_signed_forward import main as run_one
from experiments.utils import prepare_env
import os

def _noop_prepare_for_batch():
    # ensure envs created by run_one will pick up config-based action_repeat
    return


def main(runs=10, steps=1000, out_dir='experiments'):
    os.makedirs(out_dir, exist_ok=True)
    corrs = []
    sf_means = []
    a0_means = []
    for i in range(runs):
        out_path = os.path.join(out_dir, f'signed_forward_run{i+1}.json')
        print('Run', i+1, '->', out_path)
        run_one(steps=steps, out_path=out_path)
        j = json.load(open(out_path))
        recs = j.get('records_sample', [])
        if not recs:
            continue
        a = np.array([r['action'][0] for r in recs], dtype=np.float32)
        s = np.array([r['signed_forward'] for r in recs], dtype=np.float32)
        corr = float(np.corrcoef(a, s)[0,1])
        corrs.append(corr)
        sf_means.append(float(s.mean()))
        a0_means.append(float(a.mean()))
        print('  corr', corr, 'mean a0', float(a.mean()), 'mean sf', float(s.mean()))

    corrs = np.array(corrs, dtype=np.float32)
    print('SUMMARY over', len(corrs), 'runs:')
    print(' corr mean/std:', float(corrs.mean()), float(corrs.std()))
    print(' frac negative corr:', float((corrs < 0).sum())/len(corrs))
    print(' a0 mean mean/std:', float(np.mean(a0_means)), float(np.std(a0_means)))
    print(' sf mean mean/std:', float(np.mean(sf_means)), float(np.std(sf_means)))

if __name__ == '__main__':
    main(runs=10, steps=1000)
