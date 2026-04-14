#!/usr/bin/env python3
"""
Plot scatter of policy0 vs applied_forward for a given step CSV.
Usage: python scripts/plot_step_scatter.py diagnostics/slope_step74.csv
Saves: diagnostics/step74_scatter.png
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print('Usage: python scripts/plot_step_scatter.py <csv_path>')
    sys.exit(2)

csvp = Path(sys.argv[1])
if not csvp.exists():
    print('CSV not found:', csvp)
    sys.exit(2)

data = np.loadtxt(str(csvp), delimiter=',', skiprows=1)
# columns: env, policy0, applied_forward
x = data[:,1]
y = data[:,2]

slope, intercept = np.polyfit(x, y, 1)
rr = np.corrcoef(x, y)[0,1]

plt.figure(figsize=(6,6))
plt.scatter(x, y, s=30, alpha=0.8)
xs = np.linspace(np.min(x), np.max(x), 200)
plt.plot(xs, slope*xs + intercept, color='red', linewidth=2, label=f'y={slope:.3f}x+{intercept:.3f}\nr={rr:.3f}')
plt.axhline(0, color='k', linewidth=0.5)
plt.axvline(0, color='k', linewidth=0.5)
plt.xlabel('policy0')
plt.ylabel('applied_forward')
plt.title(f'Step scatter ({csvp.name})')
plt.legend()
plt.tight_layout()
out = Path('diagnostics') / f'step{csvp.stem.split("step")[-1]}_scatter.png'
plt.savefig(str(out), dpi=150)
print('Saved', out)
