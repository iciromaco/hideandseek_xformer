import os
import glob
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


def analyze_records(recs):
    a0 = np.array([r['action'][0] for r in recs], dtype=np.float32)
    sf = np.array([r['signed_forward'] for r in recs], dtype=np.float32)
    ap = np.array([r.get('applied_forward', 0.0) for r in recs], dtype=np.float32)
    # linear fits
    lin_a = np.polyfit(a0, sf, 1)
    lin_ap = np.polyfit(ap, sf, 1)
    pred_a = np.polyval(lin_a, a0)
    pred_ap = np.polyval(lin_ap, ap)
    return {
        'a0': a0,
        'applied': ap,
        'sf': sf,
        'lin_a': lin_a,
        'lin_ap': lin_ap,
        'r2_a': float(r2_score(sf, pred_a)),
        'r2_ap': float(r2_score(sf, pred_ap)),
    }


def plot_scatter(x, y, fit, title, out_path):
    plt.figure(figsize=(6,6))
    plt.scatter(x, y, s=6, alpha=0.6)
    xs = np.linspace(x.min(), x.max(), 200)
    ys = np.polyval(fit, xs)
    plt.plot(xs, ys, 'r-', lw=2)
    plt.xlabel('input')
    plt.ylabel('signed_forward')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def process_file(path, out_dir):
    j = json.load(open(path))
    recs = j.get('records_sample', [])
    if not recs:
        return None
    res = analyze_records(recs)
    base = os.path.splitext(os.path.basename(path))[0]
    out_a = os.path.join(out_dir, base + '_a0_vs_sf.png')
    out_ap = os.path.join(out_dir, base + '_applied_vs_sf.png')
    plot_scatter(res['a0'], res['sf'], res['lin_a'], base + ' a0 vs sf', out_a)
    plot_scatter(res['applied'], res['sf'], res['lin_ap'], base + ' applied vs sf', out_ap)
    summary = {
        'file': path,
        'lin_a_coeff': res['lin_a'].tolist(),
        'lin_ap_coeff': res['lin_ap'].tolist(),
        'r2_a': res['r2_a'],
        'r2_ap': res['r2_ap'],
        'mean_a0': float(res['a0'].mean()),
        'mean_applied': float(res['applied'].mean()),
        'mean_sf': float(res['sf'].mean()),
    }
    return summary


def main(glob_pattern, out_dir='experiments/plots'):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(glob_pattern))
    summaries = []
    for p in files:
        print('Processing', p)
        s = process_file(p, out_dir)
        if s:
            summaries.append(s)
            print('  r2 a0:', s['r2_a'], 'r2 applied:', s['r2_ap'])
    with open(os.path.join(out_dir, 'plot_summary.json'), 'w') as f:
        json.dump(summaries, f, indent=2)
    print('Saved plots and summary to', out_dir)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--glob', default='experiments/signed_forward_run*.json')
    ap.add_argument('--out', default='experiments/plots')
    args = ap.parse_args()
    main(args.glob, args.out)
