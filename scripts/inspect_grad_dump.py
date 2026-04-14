import torch
import sys
p = torch.load('diagnostics/grad_nan_update321_mb576_obs_encoder.0.weight.pt', map_location='cpu')
print('keys:', list(p.keys()))
for k, v in p.items():
    try:
        t = v.detach() if hasattr(v, 'detach') else v
        finite_all = bool(torch.isfinite(t).all().item())
        nan_count = int(torch.isnan(t).sum().item())
        inf_count = int(torch.isinf(t).sum().item())
        print(f"{k} shape={getattr(t,'shape',None)} dtype={getattr(t,'dtype',None)} finite_all={finite_all} nan_count={nan_count} inf_count={inf_count} min={float(t.min()):.6e} max={float(t.max()):.6e} mean={float(t.mean()):.6e}")
    except Exception as e:
        print('error reading', k, e)
