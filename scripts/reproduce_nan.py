import sys
import torch
from src.models.ppo_transformer_v3 import AgentV3

# enforce deterministic settings for replay
try:
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception as _:
    pass

fn = 'diagnostics/grad_nan_update321_mb576_obs_encoder.0.weight.pt'
if len(sys.argv) > 1:
    fn = sys.argv[1]
print('loading dump', fn)
d = torch.load(fn, map_location='cpu')
mb_obs = d.get('mb_obs')
mb_actions = d.get('mb_actions')
mb_adv = d.get('mb_adv')
state = d.get('state_dict')
print('mb_obs', None if mb_obs is None else mb_obs.shape)
print('mb_actions', None if mb_actions is None else mb_actions.shape)
print('mb_adv', None if mb_adv is None else mb_adv.shape)

# infer dims
seq_len = mb_obs.shape[1]
obs_dim = mb_obs.shape[2]
act_dim = mb_actions.shape[1]
hidden_dim = 256

device = torch.device('cpu')
agent = AgentV3(obs_dim, act_dim, hidden_dim, seq_len).to(device)
if state is not None:
    try:
        agent.load_state_dict(state)
        print('loaded state_dict into agent')
    except Exception as e:
        print('failed to load state_dict:', e)

agent.train()
mb_obs = mb_obs.to(device)
mb_actions = mb_actions.to(device)
mb_adv = mb_adv.to(device)

# follow training loss path but we lack b_logp and b_ret; use zeros/detach approximations
b_logp_dummy = torch.zeros(mb_adv.shape, device=device)
# normalize adv as training
mb_adv_n = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

from torch import autograd

try:
    with autograd.detect_anomaly():
        new_action, new_logp, entropy, new_value = agent.get_action_and_value(mb_obs, mb_actions)
        # compute surrogate
        logratio = new_logp - b_logp_dummy
        ratio = torch.exp(logratio)
        pg_loss1 = -mb_adv_n * ratio
        pg_loss2 = -mb_adv_n * torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
        v_loss = 0.0
        ent_loss = entropy.mean()
        loss = pg_loss + 0.5 * v_loss - 0.001 * ent_loss
        print('loss computed, calling backward')
        loss.backward()
        print('backward finished')
        # if dump contains optimizer state, try to step optimizer to reproduce optimizer-NaN
        opt_state = d.get('optimizer_state') or d.get('optimizer_state_dict') or d.get('opt_state_dict')
        if opt_state is not None:
            print('optimizer state found in dump — creating optimizer and loading state')
            optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3, eps=1e-5)
            try:
                optimizer.load_state_dict(opt_state)
                print('loaded optimizer state dict')
            except Exception as e:
                print('failed to load optimizer state:', e)
            try:
                optimizer.step()
                print('optimizer.step() completed')
            except Exception as e:
                print('optimizer.step() raised', e)
except Exception as e:
    import traceback
    traceback.print_exc()
    print('Exception text:', e)

# inspect gradients
for n, p in agent.named_parameters():
    if p.grad is None:
        continue
    g = p.grad
    finite = torch.isfinite(g).all().item()
    print(f'param {n} grad finite={finite} max={float(g.abs().max()):.6e} nan_count={int(torch.isnan(g).sum())}')
