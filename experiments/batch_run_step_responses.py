import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments.run_step_response_nowalls import main as run_main

cases = [ -1.0, -0.5, -0.25, 0.25, 0.5, 1.0 ]
for h in cases:
    h_tag = str(h).replace('.', 'p').replace('-', 'm')
    out_json = f'experiments/step_response_nowalls_h{h_tag}.json'
    out_png = f'experiments/plots/step_response_nowalls_h{h_tag}.png'
    print('Running hold_action', h)
    run_main(steps=400, hold_action=float(h), out_json=out_json, out_png=out_png)
    print('Saved', out_json, out_png)
print('Batch finished')
