import torch,sys
p=sys.argv[1]
try:
    x=torch.load(p,map_location='cpu')
    print('loaded',p,type(x), getattr(x,'shape',None))
    if isinstance(x, dict):
        for k,v in x.items():
            t = v.detach() if hasattr(v,'detach') else v
            try:
                print(k, getattr(t,'shape',None), 'nan', int(torch.isnan(t).sum().item()), 'inf', int(torch.isinf(t).sum().item()), 'min', float(t.min()), 'max', float(t.max()))
            except Exception as e:
                print('error reading', k, e)
    else:
        if hasattr(x,'numel'):
            print('nan_count',int(torch.isnan(x).sum().item()),'inf_count',int(torch.isinf(x).sum().item()),'min',float(x.min()),'max',float(x.max()))
except Exception as e:
    print('err',e)
