import os,torch
p='diagnostics'
for fn in os.listdir(p):
    if fn.startswith('opt_nan'):
        full=os.path.join(p,fn)
        try:
            x=torch.load(full,map_location='cpu')
            print(fn, type(x), getattr(x,'shape',None))
            if hasattr(x,'numel'):
                print('  nan',int(torch.isnan(x).sum().item()),'inf',int(torch.isinf(x).sum().item()),'min',float(x.min()),'max',float(x.max()))
        except Exception as e:
            print(fn,'err',e)
