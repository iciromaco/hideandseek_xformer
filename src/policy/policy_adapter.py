import numpy as np
import torch
from src.models.ppo_transformer_v3 import AgentV3


class PolicyAdapter:
    """小さな互換アダプタ。既存の env 内 API を置き換えつつ
    実際のポリシーロジックをここへ移譲する。
    注意: 互換性を保つため、アダプタは受け取った env オブジェクトの
    属性（例: `_inference_models`, `_policy_histories`, `shared_policy_model` 等）を更新します。
    """

    def __init__(self, env, shared_policy_model=None, shared_seq_len=8, shared_hidden_dim=128, shared_team_policy=False, shared_team_prefix=None):
        """初期化。

        env: TeamCosEnv互換オブジェクト。アダプタは env 上のポリシー関連属性を初期化・同期します。
        shared_policy_model/seq_len/hidden_dim: 共有ポリシーを使う場合の設定。
        """
        self.env = env
        # 初期状態を env と合わせる
        env._inference_models = getattr(env, '_inference_models', {})
        env._inference_seq_lens = getattr(env, '_inference_seq_lens', {})
        env._policy_histories = getattr(env, '_policy_histories', {})

        env.shared_policy_model = shared_policy_model
        env.shared_policy_seq_len = int(shared_seq_len)
        env.shared_policy_hidden_dim = int(shared_hidden_dim)
        env.shared_team_policy = bool(shared_team_policy) or (env.shared_policy_model is not None)
        env.shared_team_prefix = (
            shared_team_prefix if shared_team_prefix is not None
            else ("h" if env.learnable_agent_key.startswith("h") else "s")
        )

        self.override_learnable_policy = False
        self.model_policy_deterministic = True

    def set_shared_team_policy_state(self, state_dict, seq_len=8, hidden_dim=128):
        """共有チームポリシーの状態辞書をenvへ設定する。

        戻り値: 成功なら True, state_dict が None の場合は False を返す。
        """
        env = self.env
        if state_dict is None:
            env.shared_team_policy = False
            env.shared_policy_model = None
            return False

        env.shared_team_policy = True
        env.shared_policy_seq_len = int(seq_len)
        env.shared_policy_hidden_dim = int(hidden_dim)
        obs_dim = int(env.observation_space.shape[0])
        act_dim = int(env.action_space.shape[0])
        policy_model = AgentV3(obs_dim, act_dim, env.shared_policy_hidden_dim, env.shared_policy_seq_len)
        policy_model.load_state_dict(state_dict)
        policy_model.eval()
        env.shared_policy_model = policy_model
        for ak in env.agent_keys:
            env._policy_histories.pop((ak, env.shared_policy_seq_len), None)
        return True

    def set_inference_policy_state(self, agent_keys, state_dict, seq_len=8, hidden_dim=128):
        """指定エージェント群に対して推論モデル(state_dict)を設定する。

        agent_keys: env.agent_keys に含まれるキーのリスト。
        戻り値: 成功した場合 True。
        """
        env = self.env
        if state_dict is None:
            return False

        keys = [k for k in agent_keys if k in env.agent_keys]
        if not keys:
            return False

        obs_dim = int(env.observation_space.shape[0])
        act_dim = int(env.action_space.shape[0])
        policy_model = AgentV3(obs_dim, act_dim, int(hidden_dim), int(seq_len))
        policy_model.load_state_dict(state_dict)
        policy_model.eval()

        for ak in keys:
            env._inference_models[ak] = policy_model
            env._inference_seq_lens[ak] = int(seq_len)
            env._policy_histories.pop((ak, int(seq_len)), None)
        return True

    def set_override_learnable_policy(self, enabled):
        """学習対象ポリシーのオーバーライドを切り替える。

        True にすると学習対象は外部入力や内部モデルではなくオーバーライド動作になります。
        環境上の対応フラグも可能な限り同期します。
        """
        self.override_learnable_policy = bool(enabled)
        try:
            # 環境側のフラグも同期しておく（TeamCosEnv.step が env.override_learnable_policy を参照しているため）
            setattr(self.env, 'override_learnable_policy', bool(enabled))
        except Exception:
            pass
        return True

    def set_model_policy_deterministic(self, enabled):
        """モデルポリシーを決定論的に扱うかどうかを設定する。

        True の場合は可能な限り決定論的な推論パスを選びます。
        """
        self.model_policy_deterministic = bool(enabled)
        try:
            setattr(self.env, 'model_policy_deterministic', bool(enabled))
        except Exception:
            pass
        return True

    # --- history helpers (operate on env._policy_histories for compatibility) ---
    def _ensure_policy_history(self, agent_key, seq_len):
        """エージェント用のポリシー履歴バッファを確保・返却する。

        戻り値は履歴辞書 (buffer, ptr) 。外部から直接呼び出されることを想定。
        """
        env = self.env
        sl = int(seq_len)
        obs_dim = int(env.observation_space.shape[0])
        key = (agent_key, sl)
        hist = env._policy_histories.get(key)
        if hist is None or hist["buffer"].shape != (sl * 2, obs_dim):
            hist = {
                "buffer": np.zeros((sl * 2, obs_dim), dtype=np.float32),
                "ptr": 0,
            }
            env._policy_histories[key] = hist
        return hist

    def _prime_policy_history(self, agent_key, seq_len, norm_obs):
        """履歴バッファを現在の正規化観測で初期化（プリミング）。

        Transformer 系モデルが初期シーケンスを必要とするため呼び出す。
        """
        hist = self._ensure_policy_history(agent_key, seq_len)
        obs_np = np.asarray(norm_obs, dtype=np.float32).reshape(-1)
        sl = int(seq_len)
        for i in range(sl):
            hist["buffer"][i] = obs_np
            hist["buffer"][i + sl] = obs_np
        hist["ptr"] = 0

    def _update_policy_history(self, agent_key, seq_len, norm_obs):
        """新しい観測を履歴バッファへローテートして書き込む。"""
        hist = self._ensure_policy_history(agent_key, seq_len)
        obs_np = np.asarray(norm_obs, dtype=np.float32).reshape(-1)
        sl = int(seq_len)
        ptr = int(hist["ptr"])
        hist["buffer"][ptr] = obs_np
        hist["buffer"][ptr + sl] = obs_np
        hist["ptr"] = (ptr + 1) % sl

    def _get_policy_history_seq(self, agent_key, seq_len, norm_obs):
        """現在の履歴シーケンスを返す。履歴が空ならプリミングを行う。"""
        env = self.env
        hist = self._ensure_policy_history(agent_key, seq_len)
        if not np.any(hist["buffer"]):
            self._prime_policy_history(agent_key, seq_len, norm_obs)
            hist = self._ensure_policy_history(agent_key, seq_len)
        ptr = int(hist["ptr"])
        sl = int(seq_len)
        return hist["buffer"][ptr:ptr + sl]

    def _log_policy_source(self, agent_key, source):
        """デバッグ時にポリシー選択ソースを環境のデバッガへ渡すラッパー。"""
        # delegate to env debug logger if present
        try:
            if getattr(self.env, 'debug_mode', False):
                self.env.debug_logger.log_policy_source(agent_key, source, self.env.current_step, force=True)
        except Exception:
            pass

    def get_action(self, agent_key, norm_obs):
        """指定エージェントに対する行動を決定して返す。

        順序: per-agent model -> callable policy -> shared model -> scripted rule fallback
        常に (fwd, turn, lock, grab) のタプルを返す。
        """
        env = self.env
        # inference model per-agent
        model = env._inference_models.get(agent_key)
        if model is not None:
            self._log_policy_source(agent_key, "model")
            seq_len = int(env._inference_seq_lens.get(agent_key, 8))
            seq_np = self._get_policy_history_seq(agent_key, seq_len, norm_obs)
            seq_t = torch.as_tensor(seq_np[None, :, :], dtype=torch.float32)
            with torch.no_grad():
                if self.model_policy_deterministic and hasattr(model, "get_deterministic_action_and_value"):
                    arr = model.get_deterministic_action_and_value(seq_t)[0].cpu().numpy().reshape(-1)
                else:
                    arr = model.get_action_and_value(seq_t)[0].cpu().numpy().reshape(-1)
            try:
                # Debug: print raw model output for learnable agent on early steps
                lak = getattr(self.env, 'learnable_agent_key', None)
                cur_step = int(getattr(self.env, 'current_step', 0))
                # respect optional debug step range on env (None = no limit)
                dmin = getattr(self.env, 'debug_step_min', None)
                dmax = getattr(self.env, 'debug_step_max', None)
                in_range = True
                if dmin is not None and cur_step < int(dmin):
                    in_range = False
                if dmax is not None and cur_step > int(dmax):
                    in_range = False
                # policy debug prints removed for main28 parity
            except Exception:
                pass
            if arr.size >= 4:
                vals = (float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))
                return vals
            if arr.size >= 2:
                vals = (float(arr[0]), float(arr[1]), 0.0, 0.0)
                return vals
            raise RuntimeError(f"Invalid model action size: agent={agent_key}, size={arr.size}")

        # callable policies provided by env.inference_policies
        policy = env.inference_policies.get(agent_key)
        if policy is not None:
            try:
                self._log_policy_source(agent_key, "callable")
                pred = policy(norm_obs)
                arr = np.asarray(pred).reshape(-1)
                if arr.size >= 4:
                    vals = (float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))
                    return vals
                if arr.size >= 2:
                    vals = (float(arr[0]), float(arr[1]), 0.0, 0.0)
                    return vals
            except Exception:
                pass

        # shared team model
        if (
            env.shared_team_policy
            and env.shared_policy_model is not None
            and agent_key != env.learnable_agent_key
            and agent_key.startswith(env.shared_team_prefix)
        ):
            self._log_policy_source(agent_key, "shared_model")
            seq_np = self._get_policy_history_seq(agent_key, env.shared_policy_seq_len, norm_obs)
            seq_t = torch.as_tensor(seq_np[None, :, :], dtype=torch.float32)
            with torch.no_grad():
                if self.model_policy_deterministic and hasattr(env.shared_policy_model, "get_deterministic_action_and_value"):
                    arr = env.shared_policy_model.get_deterministic_action_and_value(seq_t)[0].cpu().numpy().reshape(-1)
                else:
                    arr = env.shared_policy_model.get_action_and_value(seq_t)[0].cpu().numpy().reshape(-1)
            if arr.size >= 4:
                vals = (float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))
                return vals
            if arr.size >= 2:
                vals = (float(arr[0]), float(arr[1]), 0.0, 0.0)
                return vals
            raise RuntimeError(f"Invalid shared action size: agent={agent_key}, size={arr.size}")

        # fallback: scripted NPC
        self._log_policy_source(agent_key, "rule")
        # If env.debug_mode is enabled, print a short diagnostic to help
        # investigate unexpected RuleBased behavior.
        arr = np.asarray(env.npcs[agent_key].get_action(norm_obs, env.idx)).reshape(-1)
        if arr.size >= 4:
            vals = (float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))
            return vals
        vals = (float(arr[0]), float(arr[1]), 0.0, 0.0)
        return vals

    # helper wrappers for callers in env
    def has_inference_model(self, agent_key):
        """エージェントに個別の推論モデルが割り当てられているかを返す。"""
        return agent_key in getattr(self.env, '_inference_models', {})

    def is_shared_enabled_for_agent(self, agent_key):
        """そのエージェントに対して共有チームポリシーが有効か判定する。"""
        env = self.env
        return bool(env.shared_team_policy and env.shared_policy_model is not None and agent_key.startswith(env.shared_team_prefix))

    def maybe_prime_for_agent(self, agent_key):
        """必要なら指定エージェントの履歴をプリミングする互換ラッパー。"""
        env = self.env
        if agent_key in env._inference_models:
            self._prime_policy_history(agent_key, int(env._inference_seq_lens.get(agent_key, 8)), self.env._normalize_obs(self.env._get_obs(self.env.agent_keys.index(agent_key))))
        if self.is_shared_enabled_for_agent(agent_key):
            self._prime_policy_history(agent_key, env.shared_policy_seq_len, self.env._normalize_obs(self.env._get_obs(self.env.agent_keys.index(agent_key))))

    def maybe_update_for_agent(self, agent_key, norm_obs_next):
        """指定エージェントの履歴バッファを次観測で更新する互換ラッパー。"""
        env = self.env
        if agent_key in env._inference_models:
            self._update_policy_history(agent_key, int(env._inference_seq_lens.get(agent_key, 8)), norm_obs_next)
        if self.is_shared_enabled_for_agent(agent_key):
            self._update_policy_history(agent_key, env.shared_policy_seq_len, norm_obs_next)
