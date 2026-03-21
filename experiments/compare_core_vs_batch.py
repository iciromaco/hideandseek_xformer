import json, os, numpy as np

ct = 'experiments/contact_trace.json'
b = 'experiments/step_response_nowalls_h0p25.json'
if not os.path.exists(ct) or not os.path.exists(b):
    print('missing files')
    raise SystemExit(1)
with open(ct,'r') as f:
    c = json.load(f)
with open(b,'r') as f:
    bb = json.load(f)
crecs = c.get('records', [])
brecs = bb.get('records', [])
cvx = np.array([r.get('vx', 0.0) for r in crecs], dtype=float)
bvx = np.array([r.get('vx', 0.0) for r in brecs], dtype=float)
L = min(len(cvx), len(bvx))
cvx = cvx[:L]
bvx = bvx[:L]
if L == 0:
    print('no data')
    raise SystemExit(1)
diff = cvx - bvx
print('len', L)
print('mean(cvx)=', cvx.mean(), 'mean(bvx)=', bvx.mean())
print('mean abs diff=', np.mean(np.abs(diff)))
print('max abs diff=', np.max(np.abs(diff)))
# Pearson
if np.std(cvx) < 1e-12 or np.std(bvx) < 1e-12:
    corr = float('nan')
else:
    corr = np.corrcoef(cvx, bvx)[0,1]
print('pearson corr=', corr)
# fraction within tol
tol = 1e-6
print('fraction within 1e-6=', np.mean(np.abs(diff) <= tol))
# print a short diff sample around 250-320
lo, hi = 250, 320
print('\nSample diffs for steps', lo, '-', hi)
for i in range(lo, min(hi+1, L)):
    print(i, cvx[i], bvx[i], cvx[i]-bvx[i])
