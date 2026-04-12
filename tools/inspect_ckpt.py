import torch,sys
p='checkpoints/HNS_seeker_s1_h2_b2_r1.pt'
try:
    obj=torch.load(p,map_location='cpu')
    print(type(obj))
    if isinstance(obj,dict):
        print(list(obj.keys())[:60])
except Exception as e:
    print('ERR',e)
