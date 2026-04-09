class PolicyManager:
    """Minimal PolicyManager helper for loading/applying policy states.

    This is intentionally small: it centralizes model state loading and
    provides `apply_to_envs` which calls either `set_shared_team_policy_state`
    or `set_inference_policy_state` depending on `shared=True`.
    """
    def __init__(self, seq_len=8, hidden_dim=128):
        self.seq_len = int(seq_len)
        self.hidden_dim = int(hidden_dim)

    def load_state(self, path):
        import torch
        try:
            return torch.load(path, map_location='cpu')
        except Exception:
            return None

    def apply_to_envs(self, env, vec_envs, agent_keys, state_dict, shared=False):
        if shared:
            # prefer set_shared_team_policy_state API
            if vec_envs is not None:
                try:
                    results = vec_envs.call('set_shared_team_policy_state', state_dict, self.seq_len, self.hidden_dim)
                    return results
                except Exception:
                    return None
            if env is not None and hasattr(env, 'set_shared_team_policy_state'):
                try:
                    # prefer policy_adapter when available
                    if hasattr(env, 'policy_adapter'):
                        return env.policy_adapter.set_shared_team_policy_state(state_dict, self.seq_len, self.hidden_dim)
                    return env.set_shared_team_policy_state(state_dict, self.seq_len, self.hidden_dim)
                except Exception:
                    return None
            return None
        else:
            # per-agent application
            if vec_envs is not None:
                try:
                    return vec_envs.call('set_inference_policy_state', list(agent_keys), state_dict, self.seq_len, self.hidden_dim)
                except Exception:
                    return None
            if env is not None and hasattr(env, 'set_inference_policy_state'):
                try:
                    if hasattr(env, 'policy_adapter'):
                        return env.policy_adapter.set_inference_policy_state(list(agent_keys), state_dict, self.seq_len, self.hidden_dim)
                    return env.set_inference_policy_state(list(agent_keys), state_dict, self.seq_len, self.hidden_dim)
                except Exception:
                    return None
            return None
