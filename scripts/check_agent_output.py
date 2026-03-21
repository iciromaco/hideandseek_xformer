#!/usr/bin/env python3
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from models.ppo_transformer_v2 import AgentV2
from src.envs.hns_environment import TeamCosEnv

env = TeamCosEnv(mode="initial", target="seeker", n_seekers=1, n_hiders=2, render_mode=None)
obs, _ = env.reset()
obs_np = np.asarray(obs, dtype=np.float32).reshape(-1)
# Agent expects input shape (batch, seq_len, obs_dim). Repeat single obs to form a sequence.
seq_len = 16
obs_seq = np.repeat(obs_np[None, :], seq_len, axis=0)[None, :, :]
obs_t = torch.as_tensor(obs_seq, dtype=torch.float32)
agent = AgentV2(obs_t.shape[-1], env.action_space.shape[0], 256, 16)
path = os.path.join(
    "checkpoints",
    f"HNS_V27_seeker_s{env.n_seekers}_h{env.n_hiders}_b{env.n_boxes}_r{env.n_ramps}.pt",
)
print("checkpoint path:", path)
if os.path.exists(path):
    agent.load_state_dict(torch.load(path, map_location="cpu"))
    agent.eval()
    with torch.no_grad():
        out = agent.get_action_and_value(obs_t)[0].cpu().numpy()
    print("agent output (first 8):", out.reshape(-1)[:8])
else:
    print("checkpoint not found:", path)
env.close()
